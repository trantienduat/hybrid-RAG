"""
FalkorDB graph client: upsert nodes and edges from ParseResult/Triples.

Uses redis-py (FalkorDB uses Redis protocol) via the `falkordb` driver.
Graph name: configured via FALKORDB_GRAPH env var (default: "codebase").

Node merge key:  id property
Edge merge: src_id + rel + dst_id (no duplicate edges for same relationship).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import falkordb

from hybrid_rag.ingestion.parser import NodeData, ParseResult
from hybrid_rag.ingestion.triplet_extractor import Triple, extract_triples

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 6379
_DEFAULT_GRAPH = "codebase"


class GraphClient:
    """Thin wrapper around FalkorDB for KG upsert operations."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        graph_name: str | None = None,
    ) -> None:
        self._host = host or os.environ.get("FALKORDB_HOST", _DEFAULT_HOST)
        self._port = int(port or os.environ.get("FALKORDB_PORT", _DEFAULT_PORT))
        self._graph_name = graph_name or os.environ.get("FALKORDB_GRAPH", _DEFAULT_GRAPH)
        self._db = falkordb.FalkorDB(host=self._host, port=self._port)
        self._graph = self._db.select_graph(self._graph_name)
        logger.info("GraphClient connected: %s:%d graph=%s", self._host, self._port, self._graph_name)

    # ── Public API ────────────────────────────────────────────────

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

    # ── Internal ──────────────────────────────────────────────────

    def _upsert_nodes(self, nodes: list[NodeData]) -> int:
        count = 0
        for node in nodes:
            props = _sanitize(node.properties)
            props["id"] = node.id
            cypher = (
                f"MERGE (n:{node.label} {{id: $id}}) "
                f"SET n += $props"
            )
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
                # Log and continue — missing stub nodes are expected for call stubs
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
