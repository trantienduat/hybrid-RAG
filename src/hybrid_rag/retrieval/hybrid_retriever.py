"""
Hybrid Retriever — implements BaseRetriever using graph + vector + RRF fusion.

Pipeline:
  1. QueryAnalyzer  — classify query as structural / semantic / hybrid
  2. GraphRetriever — run Cypher node/neighbor queries  (structural + hybrid)
  3. VectorRetriever — run embedding similarity search  (always)
  4. RRF             — fuse both ranked lists
  5. Return top_k merged results

Depends only on ports (GraphStore, VectorStore, BaseEmbedder) — fully
vendor-neutral. Inject FalkorDBStore / QdrantStore / OllamaEmbedder at the
composition root (cli.py).
"""
from __future__ import annotations

import logging
from typing import Any

from hybrid_rag.ports.embedder import BaseEmbedder
from hybrid_rag.ports.graph_store import GraphStore
from hybrid_rag.ports.retriever import BaseRetriever
from hybrid_rag.ports.vector_store import VectorStore
from hybrid_rag.retrieval.context_assembler import ContextAssembler, RetrievalContext
from hybrid_rag.retrieval.graph_retriever import GraphRetriever
from hybrid_rag.retrieval.query_analyzer import analyze
from hybrid_rag.retrieval.rrf import reciprocal_rank_fusion
from hybrid_rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Graph + Vector hybrid retriever fused via Reciprocal Rank Fusion.

    - Structural queries: graph retrieval + vector retrieval → RRF
    - Semantic queries:   vector retrieval only → RRF (single list)
    - Hybrid queries:     both → RRF
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        rrf_k: int = 60,
        rrf_structural_weight: float = 3.0,
        rrf_hybrid_weight: float = 1.5,
    ) -> None:
        self._graph_retriever = GraphRetriever(graph_store)
        self._vector_retriever = VectorRetriever(vector_store, embedder)
        self._assembler = ContextAssembler()
        self._rrf_k = rrf_k
        self._rrf_structural_weight = rrf_structural_weight
        self._rrf_hybrid_weight = rrf_hybrid_weight

    # ── BaseRetriever interface ────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        skip_graph: bool = False,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run hybrid retrieval and return top_k fused results sorted by rrf_score.

        Each result dict has at least:
          node_id, base_node_id, name, label, file_path,
          text, rel, source, rrf_score.

        Args:
            skip_graph: When True, skip graph retrieval entirely (vector-only
                        baseline for evaluation / ablation studies).
            repository: Custom repository namespace to filter results by.
        """
        analysis = analyze(query)
        logger.debug("HybridRetriever query analysis: %s", analysis)

        graph_results: list[dict[str, Any]] = []
        if not skip_graph and analysis.query_type in ("structural", "hybrid"):
            graph_results = self._graph_retriever.retrieve(analysis, top_k=top_k * 2, repository=repository)
            logger.debug("Graph results: %d nodes", len(graph_results))

        filter_payload = {"repository": repository} if repository else None
        vector_results = self._vector_retriever.retrieve(query, top_k=top_k * 2, filter_payload=filter_payload)
        logger.debug("Vector results: %d chunks", len(vector_results))

        # Structural queries are relationship/structure lookups — graph evidence
        # should dominate so that structural nodes (which have no vector text)
        # aren't outranked by semantically-similar but irrelevant vector chunks.
        if skip_graph:
            rrf_weights = (0.0, 1.0)
        elif analysis.query_type == "structural":
            rrf_weights = (self._rrf_structural_weight, 1.0)
        elif analysis.query_type == "hybrid":
            rrf_weights = (self._rrf_hybrid_weight, 1.0)
        else:
            rrf_weights = (1.0, 1.0)

        fused = reciprocal_rank_fusion(
            graph_results, vector_results, k=self._rrf_k, weights=rrf_weights
        )
        return fused[:top_k]

    def close(self) -> None:
        self._vector_retriever._embedder.close()

    # ── Convenience ───────────────────────────────────────────────

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 20,
        max_tokens: int | None = 2048,
        max_chars: int | None = None,
        context_n: int | None = None,
        repository: str | None = None,
    ) -> RetrievalContext:
        """Retrieve and assemble context in one call with dynamic token/char budgeting and scoping."""
        results = self.retrieve(query, top_k=top_k, repository=repository)

        # Fallback to legacy top_n count assembly if budget is explicitly omitted and legacy count is provided
        if max_tokens is None and max_chars is None and context_n is not None:
            return self._assembler.assemble(results, top_n=context_n, query=query)

        # Pack context dynamically under budget constraints
        return self._assembler.assemble(
            results,
            max_tokens=max_tokens,
            max_chars=max_chars,
            query=query,
        )
