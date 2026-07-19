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
        results: list[dict[str, Any]] = []

        # Search by entities only — keywords are generic terms (e.g. "classes",
        # "inherit") that produce noisy graph matches; vector search handles them.
        search_terms = list(dict.fromkeys(analysis.entities))

        for term in search_terms:
            if len(results) >= top_k:
                break
            nodes = self._store.find_nodes(term, limit=top_k)
            for node in nodes:
                # Apply repository filter if provided
                if repository and node.get("repository") != repository:
                    continue
                nid = node.get("node_id", "")
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    results.append(_node_to_result(node))

        # For non-global (local) queries, expand direct neighbors of seed nodes.
        # Query in-bound and out-bound separately so each direction gets its own
        # slot budget — prevents outgoing DEFINES edges (methods) from drowning
        # out incoming INHERITS edges (subclasses / callers) when limit is shared.
        if analysis.query_type != "global":
            seed_ids = [r["node_id"] for r in results[:_NEIGHBOR_SEEDS]]
            for seed_id in seed_ids:
                if len(results) >= top_k * 4:
                    break
                for direction in ("in", "out"):
                    neighbors = self._store.find_neighbors(
                        seed_id, direction=direction, max_hops=1, limit=_NEIGHBOR_LIMIT
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
