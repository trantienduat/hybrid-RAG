"""
Port: GraphStore — abstract interface for knowledge graph operations.

Concrete adapters:
  hybrid_rag.graph.falkordb_store.FalkorDBStore

Any future graph backend (Neo4j, Memgraph, Amazon Neptune…) must implement
this interface. Business logic depends only on GraphStore, never on a vendor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hybrid_rag.ingestion.parser import ParseResult


class GraphStore(ABC):
    """Contract for knowledge-graph backends."""

    @abstractmethod
    def ingest(self, result: ParseResult) -> dict[str, int]:
        """
        Upsert all nodes and edges from a ParseResult.

        Returns {"nodes": <count>, "edges": <count>}.
        """

    @abstractmethod
    def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a read query and return raw results."""

    @abstractmethod
    def node_count(self) -> int:
        """Return total number of nodes in the graph."""

    @abstractmethod
    def edge_count(self) -> int:
        """Return total number of edges in the graph."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all nodes and edges. Intended for tests only."""
