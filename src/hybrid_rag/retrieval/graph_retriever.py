"""
Graph Retriever — executes structured Cypher queries against the knowledge graph.

Given a QueryAnalysis, searches for matching nodes by name and (for structural
or hybrid queries) expands to their direct neighbors to capture relationship context.

Depends only on the GraphStore port — fully vendor-neutral.
"""

from __future__ import annotations

import logging
from typing import Any

from hybrid_rag.ports.graph_store import GraphStore
from hybrid_rag.retrieval.query_analyzer import QueryAnalysis

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 20
_NEIGHBOR_SEEDS = 10  # expand neighbors from top N matched nodes
_NEIGHBOR_LIMIT = 30  # neighbors per seed per direction (in + out queried separately)


class GraphRetriever:
    """Retrieve relevant graph nodes and structural context for a query."""

    def __init__(self, graph_store: GraphStore) -> None:
        self._store = graph_store

    def retrieve(
        self,
        analysis: QueryAnalysis,
        top_k: int = _DEFAULT_TOP_K,
        repository: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return up to top_k graph nodes relevant to the analysis.

        Each result dict has:
          node_id, base_node_id, name, label, file_path, rel, text, source.
        """
        if not analysis.entities and not analysis.keywords:
            return []

        seen_ids: set[str] = set()
        seeds: list[dict[str, Any]] = []

        # Search by entities only — keywords are generic terms (e.g. "classes",
        # "inherit") that produce noisy graph matches; vector search handles them.
        search_terms = list(dict.fromkeys(analysis.entities))

        for term in search_terms:
            if len(seeds) >= top_k:
                break
            nodes = self._store.find_nodes(term, repository=repository, limit=top_k)
            for node in nodes:
                nid = node.get("node_id", "")
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    seeds.append(_node_to_result(node))

        results: list[dict[str, Any]] = []
        if analysis.query_type != "global":
            seed_ids = [r["node_id"] for r in seeds[:_NEIGHBOR_SEEDS]]
            directions = [analysis.direction] if analysis.direction else ["in", "out"]
            for seed_id in seed_ids:
                if len(results) >= top_k * 4:
                    break
                for direction in directions:
                    neighbors = self._store.find_neighbors(
                        seed_id,
                        rel=analysis.relation,
                        direction=direction,
                        max_hops=analysis.max_hops,
                        limit=_NEIGHBOR_LIMIT,
                    )
                    for nb in neighbors:
                        # Apply repository filter if provided
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
                                }
                            )

        # For relation questions, targets are the answer and should rank before
        # the matching seed. General searches retain seed-first behavior.
        results = results + seeds if analysis.relation else seeds + results

        logger.debug(
            "GraphRetriever: %d results for %r (scoped to repo=%s)",
            len(results),
            analysis,
            repository,
        )
        return results[:top_k]


def _node_to_result(node: dict[str, Any]) -> dict[str, Any]:
    nid = node.get("node_id", "")
    return {
        "node_id": nid,
        "base_node_id": nid,
        "name": node.get("name", ""),
        "label": node.get("label", ""),
        "file_path": node.get("file_path", ""),
        "repository": node.get("repository", ""),
        "rel": "",
        "text": "",
        "source": "graph",
    }
