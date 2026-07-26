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
from concurrent.futures import ThreadPoolExecutor
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
        rrf_structural_weight: float = 1.5,  # acts as the unified graph weight
        rrf_hybrid_weight: float = 1.5,  # unused, kept for compatibility
    ) -> None:
        self._graph_store = graph_store
        self._graph_retriever = GraphRetriever(graph_store)
        self._vector_retriever = VectorRetriever(vector_store, embedder)
        self._assembler = ContextAssembler()
        self._rrf_k = rrf_k
        self._rrf_graph_weight = rrf_structural_weight
        self._last_timings: dict[str, float] = {}

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
        import time

        self._last_timings = {
            "parse_query_ms": 0.0,
            "graph_search_ms": 0.0,
            "vector_search_ms": 0.0,
            "rrf_ms": 0.0,
        }

        t_start = time.perf_counter()
        analysis = analyze(query)
        self._last_timings["parse_query_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        logger.debug("HybridRetriever query analysis: %s", analysis)

        # ── Query Routing (Milestone 3.3) ──
        import os

        query_routing_env = os.environ.get("QUERY_ROUTING", "false").lower() == "true"
        filter_payload = {}
        if repository:
            filter_payload["repository"] = repository
        if query_routing_env:
            non_code_keywords = {
                "docker",
                "yaml",
                "yml",
                "readme",
                "config",
                "compose",
                "port",
                "host",
                "k8s",
                "deploy",
                "setup",
                "install",
            }
            is_non_code_query = any(kw in query.lower() for kw in non_code_keywords)
            filter_payload["file_type"] = "non_code" if is_non_code_query else "code"

        # ── Vector Routing (Layer 2) ──
        vector_routing_env = os.environ.get("VECTOR_ROUTING", "false").lower() == "true"
        routing_fallback_env = os.environ.get("ROUTING_FALLBACK", "true").lower() == "true"
        routing_threshold_env = float(os.environ.get("ROUTING_THRESHOLD", "0.70"))
        routing_anchors_env = int(os.environ.get("ROUTING_ANCHORS", "6"))

        if not skip_graph and analysis.query_type == "local" and vector_routing_env:
            t_vector = time.perf_counter()
            vector_results = self._vector_retriever.retrieve(
                query, top_k=top_k * 2, filter_payload=filter_payload
            )
            self._last_timings["vector_search_ms"] = round(
                (time.perf_counter() - t_vector) * 1000, 2
            )

            best_score = vector_results[0].get("score", 0.0) if vector_results else 0.0
            if routing_fallback_env and best_score < routing_threshold_env:
                logger.debug(
                    "Vector Routing similarity %.3f below threshold %.3f. Falling back to Parallel.",
                    best_score,
                    routing_threshold_env,
                )
            else:
                t_graph = time.perf_counter()
                anchors = vector_results[:routing_anchors_env]
                seen_ids = set()
                results = []

                # Add anchor nodes themselves
                for anchor in anchors:
                    bid = anchor.get("base_node_id", "")
                    if bid and bid not in seen_ids:
                        seen_ids.add(bid)
                        results.append(
                            {
                                "node_id": bid,
                                "base_node_id": bid,
                                "name": anchor.get("name") or bid.split("::")[-1],
                                "label": anchor.get("label", "Unknown"),
                                "file_path": anchor.get("file_path", ""),
                                "repository": anchor.get("repository", ""),
                                "rel": "",
                                "text": anchor.get("text", ""),
                                "source": "vector",
                                "rrf_score": anchor.get("score", 1.0),
                            }
                        )

                # Expand structural neighbors
                for anchor in list(results):
                    seed_id = anchor["node_id"]
                    for direction in ("in", "out"):
                        try:
                            neighbors = self._graph_store.find_neighbors(
                                seed_id, direction=direction, max_hops=1, limit=10
                            )
                            for nb in neighbors:
                                if repository and nb.get("dst_repository") != repository:
                                    continue
                                nid = nb.get("dst_id", "")
                                if nid and nid not in seen_ids:
                                    seen_ids.add(nid)
                                    results.append(
                                        {
                                            "node_id": nid,
                                            "base_node_id": nid,
                                            "name": nb.get("dst_name", ""),
                                            "label": nb.get("dst_label", ""),
                                            "file_path": nb.get("dst_file_path", ""),
                                            "repository": nb.get("dst_repository", ""),
                                            "rel": nb.get("rel", ""),
                                            "text": "",
                                            "source": "graph",
                                            "rrf_score": anchor["rrf_score"] * 0.9,
                                        }
                                    )
                        except Exception as exc:
                            logger.warning(
                                "Vector Routing neighbor expansion failed for %s: %s", seed_id, exc
                            )

                self._last_timings["graph_search_ms"] = round(
                    (time.perf_counter() - t_graph) * 1000, 2
                )
                results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
                return results[:top_k]

        if not skip_graph and analysis.query_type == "global":
            # Global query: fetch community summaries, optionally scoped by repository.
            t_graph = time.perf_counter()
            try:
                if repository:
                    cypher = (
                        "MATCH (c:Community)<-[:IN_COMMUNITY]-(n) "
                        "WHERE n.repository = $repository "
                        "RETURN DISTINCT c.id AS id, c.name AS name, c.summary AS summary"
                    )
                    res = self._graph_store.query(cypher, {"repository": repository})
                else:
                    cypher = (
                        "MATCH (c:Community) "
                        "RETURN c.id AS id, c.name AS name, c.summary AS summary"
                    )
                    res = self._graph_store.query(cypher)
                global_results = []
                for row in res.result_set or []:
                    comm_id, name, summary = row[0], row[1], row[2]
                    global_results.append(
                        {
                            "node_id": comm_id,
                            "base_node_id": comm_id,
                            "name": name,
                            "label": "Community",
                            "file_path": "",
                            "rel": "",
                            "text": f"### Community Component: {name}\n\n{summary}",
                            "source": "graph",
                            "rrf_score": 1.0,
                        }
                    )
                self._last_timings["graph_search_ms"] = round(
                    (time.perf_counter() - t_graph) * 1000, 2
                )
                logger.debug("Retrieved %d communities for global query", len(global_results))
                if global_results:
                    return global_results
            except Exception as exc:
                self._last_timings["graph_search_ms"] = round(
                    (time.perf_counter() - t_graph) * 1000, 2
                )
                logger.error("Failed to retrieve communities for global query: %s", exc)

        graph_results: list[dict[str, Any]] = []

        # Exact structural relations are cheaper and more precise through the
        # graph. Fall back to vector retrieval only when the graph has no answer.
        if not skip_graph and analysis.query_type == "local" and analysis.relation:
            t_graph = time.perf_counter()
            graph_results = self._graph_retriever.retrieve(
                analysis, top_k=top_k * 2, repository=repository
            )
            self._last_timings["graph_search_ms"] = round((time.perf_counter() - t_graph) * 1000, 2)
            logger.debug("Graph results: %d nodes", len(graph_results))
            has_relation_target = any(
                result.get("result_role") == "relation_target" for result in graph_results
            )
            if has_relation_target:
                t_rrf = time.perf_counter()
                fused = reciprocal_rank_fusion(
                    graph_results,
                    [],
                    k=self._rrf_k,
                    weights=(self._rrf_graph_weight, 0.0),
                )
                self._last_timings["rrf_ms"] = round((time.perf_counter() - t_rrf) * 1000, 2)
                return fused[:top_k]
            graph_results = []

        def retrieve_graph() -> tuple[list[dict[str, Any]], float]:
            started = time.perf_counter()
            found = self._graph_retriever.retrieve(
                analysis,
                top_k=top_k * 2,
                repository=repository,
            )
            return found, round((time.perf_counter() - started) * 1000, 2)

        def retrieve_vector() -> tuple[list[dict[str, Any]], float]:
            started = time.perf_counter()
            clean_semantic_candidates = (
                not skip_graph and analysis.relation is None and "__call__" not in query
            )
            found = self._vector_retriever.retrieve(
                query,
                top_k=top_k * (4 if clean_semantic_candidates else 2),
                filter_payload=filter_payload,
            )
            if clean_semantic_candidates:
                canonical = []
                placeholders = []
                for result in found:
                    is_placeholder = any(
                        str(result.get(field, "")).startswith("__call__")
                        for field in ("name", "node_id", "base_node_id")
                    )
                    (placeholders if is_placeholder else canonical).append(result)
                if analysis.target_label:
                    matching = [
                        result
                        for result in canonical
                        if result.get("label") == analysis.target_label
                    ]
                    if matching:
                        canonical = matching + [
                            result
                            for result in canonical
                            if result.get("label") != analysis.target_label
                        ]
                found = (canonical + placeholders)[: top_k * 2]
            return found, round((time.perf_counter() - started) * 1000, 2)

        should_search_graph = (
            not skip_graph and analysis.query_type == "local" and not analysis.relation
        )
        if should_search_graph:
            with ThreadPoolExecutor(max_workers=2) as executor:
                graph_future = executor.submit(retrieve_graph)
                vector_future = executor.submit(retrieve_vector)
                graph_results, self._last_timings["graph_search_ms"] = graph_future.result()
                vector_results, self._last_timings["vector_search_ms"] = vector_future.result()
        else:
            vector_results, self._last_timings["vector_search_ms"] = retrieve_vector()

        logger.debug("Graph results: %d nodes", len(graph_results))
        logger.debug("Vector results: %d chunks", len(vector_results))

        t_rrf = time.perf_counter()
        if skip_graph:
            rrf_weights = (0.0, 1.0)
        else:
            # Generic graph expansion is noisier than an explicit relation
            # traversal; do not let it overpower a strong vector ranking.
            rrf_weights = (min(self._rrf_graph_weight, 1.0), 1.0)

        fused = reciprocal_rank_fusion(
            graph_results, vector_results, k=self._rrf_k, weights=rrf_weights
        )
        self._last_timings["rrf_ms"] = round((time.perf_counter() - t_rrf) * 1000, 2)
        return fused[:top_k]

    def close(self) -> None:
        self._vector_retriever.clear_cache()
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

        function_ids = [
            item.get("base_node_id") or item.get("node_id", "")
            for item in results
            if item.get("label") == "Function" and item.get("node_id")
        ]
        siblings_by_id = self._get_called_siblings_batch(function_ids)

        # Attach called sibling names for context skeletonization.
        for item in results:
            node_id = item.get("base_node_id") or item.get("node_id", "")
            siblings = siblings_by_id.get(node_id, [])
            if siblings:
                item["called_siblings"] = siblings

        # Fallback to legacy top_n count assembly if budget is explicitly omitted and legacy count is provided
        if max_tokens is None and max_chars is None and context_n is not None:
            ctx = self._assembler.assemble(results, top_n=context_n, query=query)
        else:
            # Pack context dynamically under budget constraints
            ctx = self._assembler.assemble(
                results,
                max_tokens=max_tokens,
                max_chars=max_chars,
                query=query,
            )
        ctx.timings = self._last_timings.copy()
        return ctx

    def _get_called_siblings_batch(self, node_ids: list[str]) -> dict[str, list[str]]:
        """Fetch called sibling names for all result functions in one query."""
        if not node_ids:
            return {}
        cypher = (
            "UNWIND $ids AS id "
            "MATCH (f:Function {id: id})-[:CALLS]->(sibling:Function) "
            "WHERE sibling.file_path = f.file_path "
            "RETURN id, collect(DISTINCT sibling.name) AS names"
        )
        try:
            res = self._graph_store.query(cypher, {"ids": node_ids})
            siblings_by_id: dict[str, list[str]] = {}
            for row in res.result_set or []:
                siblings_by_id[str(row[0])] = [str(name) for name in (row[1] or [])]
            return siblings_by_id
        except Exception as exc:
            logger.warning("Failed to fetch called siblings: %s", exc)
            return {}
