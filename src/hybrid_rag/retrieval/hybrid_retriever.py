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
    ) -> None:
        self._graph_retriever = GraphRetriever(graph_store)
        self._vector_retriever = VectorRetriever(vector_store, embedder)
        self._assembler = ContextAssembler()
        self._rrf_k = rrf_k

    # ── BaseRetriever interface ────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Run hybrid retrieval and return top_k fused results sorted by rrf_score.

        Each result dict has at least:
          node_id, base_node_id, name, label, file_path,
          text, rel, source, rrf_score.
        """
        analysis = analyze(query)
        logger.debug("HybridRetriever query analysis: %s", analysis)

        graph_results: list[dict[str, Any]] = []
        if analysis.query_type in ("structural", "hybrid"):
            graph_results = self._graph_retriever.retrieve(analysis, top_k=top_k * 2)
            logger.debug("Graph results: %d nodes", len(graph_results))

        vector_results = self._vector_retriever.retrieve(query, top_k=top_k * 2)
        logger.debug("Vector results: %d chunks", len(vector_results))

        fused = reciprocal_rank_fusion(graph_results, vector_results, k=self._rrf_k)
        return fused[:top_k]

    def close(self) -> None:
        self._vector_retriever._embedder.close()

    # ── Convenience ───────────────────────────────────────────────

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 10,
        context_n: int = 5,
    ) -> RetrievalContext:
        """Retrieve and assemble context in one call."""
        results = self.retrieve(query, top_k=top_k)
        return self._assembler.assemble(results, top_n=context_n, query=query)
