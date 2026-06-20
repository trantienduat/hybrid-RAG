"""
FalkorDB adapter — implements GraphStore port.

Vendor: FalkorDB (Redis-protocol graph database).
Swap this file for a different adapter (e.g. neo4j_store.py) to change backends.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import falkordb

from hybrid_rag.ingestion.parser import NodeData, ParseResult
from hybrid_rag.ingestion.triplet_extractor import Triple, extract_triples
from hybrid_rag.ports.graph_store import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 6379
_DEFAULT_GRAPH = "codebase"


class FalkorDBStore(GraphStore):
    """GraphStore adapter backed by FalkorDB."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        graph_name: str | None = None,
    ) -> None:
        self._host = host or os.environ.get("FALKORDB_HOST") or _DEFAULT_HOST
        self._port = int(port or os.environ.get("FALKORDB_PORT") or _DEFAULT_PORT)
        self._graph_name = graph_name or os.environ.get("FALKORDB_GRAPH") or _DEFAULT_GRAPH
        self._db = falkordb.FalkorDB(host=self._host, port=self._port)
        self._graph = self._db.select_graph(self._graph_name)
        logger.info(
            "FalkorDBStore connected: %s:%d graph=%s", self._host, self._port, self._graph_name
        )

    # ── GraphStore interface ──────────────────────────────────────

    def ingest(self, result: ParseResult) -> dict[str, int]:
        """Upsert all nodes and edges from a ParseResult. Returns counts."""
        node_count = self._upsert_nodes(result.nodes)
        triples = extract_triples(result)
        edge_count = self._upsert_edges(triples)
        logger.info("Ingested %d nodes, %d edges", node_count, edge_count)
        return {"nodes": node_count, "edges": edge_count}

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Run arbitrary read query."""
        return self._graph.query(cypher, params or {})

    def node_count(self) -> int:
        res = self._graph.query("MATCH (n) RETURN count(n) AS c")
        return res.result_set[0][0] if res.result_set else 0

    def edge_count(self) -> int:
        res = self._graph.query("MATCH ()-[r]->() RETURN count(r) AS c")
        return res.result_set[0][0] if res.result_set else 0

    def clear(self) -> None:
        """Delete all nodes and edges. Use in tests only."""
        self._graph.query("MATCH (n) DETACH DELETE n")

    def find_nodes(
        self,
        name: str,
        label: str | None = None,
        repository: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find nodes whose name contains *name* (case-sensitive substring)."""
        if not name:
            return []

        where_clauses = ["n.name CONTAINS $name"]
        params = {"name": name}
        if repository:
            where_clauses.append("n.repository = $repository")
            params["repository"] = repository

        where_str = " AND ".join(where_clauses)
        if label:
            cypher = f"MATCH (n:{label}) WHERE {where_str} RETURN n LIMIT {limit}"
        else:
            cypher = f"MATCH (n) WHERE {where_str} RETURN n LIMIT {limit}"

        res = self._graph.query(cypher, params)
        results: list[dict[str, Any]] = []
        for row in res.result_set or []:
            node = row[0]
            props = getattr(node, "properties", {})
            labels = getattr(node, "labels", [])
            results.append(
                {
                    "node_id": props.get("id", ""),
                    "label": labels[0] if labels else "Unknown",
                    "name": props.get("name", ""),
                    "file_path": props.get("file_path", ""),
                    "repository": props.get("repository", ""),
                }
            )
        return results

    def list_repositories(self) -> list[str]:
        """Return a list of all unique repository names in the graph."""
        res = self._graph.query(
            "MATCH (n) WHERE n.repository IS NOT NULL AND n.repository <> '' RETURN DISTINCT n.repository AS repo"
        )
        return [str(row[0]) for row in (res.result_set or []) if row[0]]

    def find_neighbors(
        self,
        node_id: str,
        rel: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Find nodes connected to *node_id*."""
        rel_filter = f"[r:{rel}]" if rel else "[r]"

        if max_hops > 1:
            # Variable-length path: return reachable endpoint nodes only
            hops = f"*1..{max_hops}"
            if direction == "out":
                cypher = f"MATCH (n {{id: $id}})-[{hops}]->(m) RETURN m LIMIT {limit}"
            elif direction == "in":
                cypher = f"MATCH (m)-[{hops}]->(n {{id: $id}}) RETURN m LIMIT {limit}"
            else:
                cypher = f"MATCH (n {{id: $id}})-[{hops}]-(m) RETURN m LIMIT {limit}"
            res = self._graph.query(cypher, {"id": node_id})
            results: list[dict[str, Any]] = []
            for row in res.result_set or []:
                m_node = row[0]
                m_props = getattr(m_node, "properties", {})
                m_labels = getattr(m_node, "labels", [])
                results.append(
                    {
                        "src_id": node_id,
                        "rel": "REACHABLE",
                        "dst_id": m_props.get("id", ""),
                        "dst_label": m_labels[0] if m_labels else "Unknown",
                        "dst_name": m_props.get("name", ""),
                        "dst_file_path": m_props.get("file_path", ""),
                        "dst_repository": m_props.get("repository", ""),
                    }
                )
            return results

        # Single-hop: include relationship type
        if direction == "out":
            cypher = (
                f"MATCH (n {{id: $id}})-{rel_filter}->(m) "
                f"RETURN type(r) AS rel_type, m LIMIT {limit}"
            )
        elif direction == "in":
            cypher = (
                f"MATCH (m)-{rel_filter}->(n {{id: $id}}) "
                f"RETURN type(r) AS rel_type, m LIMIT {limit}"
            )
        else:
            cypher = (
                f"MATCH (n {{id: $id}})-{rel_filter}-(m) "
                f"RETURN type(r) AS rel_type, m LIMIT {limit}"
            )
        res = self._graph.query(cypher, {"id": node_id})
        results = []
        for row in res.result_set or []:
            rel_str, m_node = row[0], row[1]
            m_props = getattr(m_node, "properties", {})
            m_labels = getattr(m_node, "labels", [])
            results.append(
                {
                    "src_id": node_id,
                    "rel": rel_str,
                    "dst_id": m_props.get("id", ""),
                    "dst_label": m_labels[0] if m_labels else "Unknown",
                    "dst_name": m_props.get("name", ""),
                    "dst_file_path": m_props.get("file_path", ""),
                    "dst_repository": m_props.get("repository", ""),
                }
            )
        return results

    def delete_file_nodes(self, file_path: str, repository: str) -> None:
        """Delete all nodes associated with a specific file in a repository."""
        cypher = (
            "MATCH (n) "
            "WHERE n.file_path = $file_path AND n.repository = $repository "
            "DETACH DELETE n"
        )
        self._graph.query(cypher, {"file_path": file_path, "repository": repository})

    def get_repository_commit(self, repository: str) -> str | None:
        """Retrieve the last indexed commit hash for a repository."""
        cypher = (
            "MATCH (r:RepositoryMetadata {id: $repo}) "
            "RETURN r.last_indexed_commit AS commit"
        )
        res = self._graph.query(cypher, {"repo": repository})
        if res.result_set:
            return res.result_set[0][0]
        return None

    def set_repository_commit(self, repository: str, commit_hash: str, repo_path: str | None = None) -> None:
        """Save the last indexed commit hash for a repository."""
        if repo_path:
            cypher = (
                "MERGE (r:RepositoryMetadata {id: $repo}) "
                "SET r.last_indexed_commit = $commit_hash, r.repo_path = $repo_path, r.updated_at = timestamp()"
            )
            self._graph.query(cypher, {"repo": repository, "commit_hash": commit_hash, "repo_path": repo_path})
        else:
            cypher = (
                "MERGE (r:RepositoryMetadata {id: $repo}) "
                "SET r.last_indexed_commit = $commit_hash, r.updated_at = timestamp()"
            )
            self._graph.query(cypher, {"repo": repository, "commit_hash": commit_hash})

    def get_repository_metadata(self, repository: str) -> dict[str, Any] | None:
        """Retrieve repository metadata including last indexed commit, path, and last synced time."""
        cypher = (
            "MATCH (r:RepositoryMetadata {id: $repo}) "
            "RETURN r.last_indexed_commit AS commit, r.repo_path AS path, r.updated_at AS updated_at"
        )
        res = self._graph.query(cypher, {"repo": repository})
        if res.result_set:
            return {
                "last_indexed_commit": res.result_set[0][0],
                "repo_path": res.result_set[0][1],
                "updated_at": res.result_set[0][2]
            }
        return None

    # ── Internal ──────────────────────────────────────────────────

    def _upsert_nodes(self, nodes: list[NodeData]) -> int:
        count = 0
        for node in nodes:
            props = _sanitize(node.properties)
            props["id"] = node.id
            cypher = f"MERGE (n:{node.label} {{id: $id}}) SET n += $props"
            self._graph.query(cypher, {"id": node.id, "props": props})
            count += 1
        return count

    def _upsert_edges(self, triples: list[Triple]) -> int:
        count = 0
        for t in triples:
            props = _sanitize(t.properties)
            cypher = (
                f"MATCH (a:{t.src_label} {{id: $src}}) "
                f"MATCH (b:{t.dst_label} {{id: $dst}}) "
                f"MERGE (a)-[r:{t.rel}]->(b) "
                f"SET r += $props"
            )
            try:
                self._graph.query(cypher, {"src": t.src_id, "dst": t.dst_id, "props": props})
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("Edge skipped %s->%s [%s]: %s", t.src_id, t.dst_id, t.rel, exc)
        return count


def _sanitize(props: dict[str, Any]) -> dict[str, Any]:
    """Convert values to FalkorDB-safe types (no None, no complex objects)."""
    out: dict[str, Any] = {}
    for k, v in props.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out
