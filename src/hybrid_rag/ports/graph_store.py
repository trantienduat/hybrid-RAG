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

    @abstractmethod
    def find_nodes(
        self,
        name: str,
        label: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Find nodes whose name property matches (substring).

        Returns list of dicts with keys: node_id, label, name, file_path.
        """

    @abstractmethod
    def find_neighbors(
        self,
        node_id: str,
        rel: str | None = None,
        direction: str = "out",
        max_hops: int = 1,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Find nodes connected to node_id.

        direction: "out" (n→m), "in" (m→n), "both" (undirected).
        max_hops: 1 returns direct neighbors with rel type; >1 returns reachable
          endpoints (rel set to "REACHABLE").
        Returns list of dicts: src_id, rel, dst_id, dst_label, dst_name, dst_file_path.
        """

    @abstractmethod
    def delete_file_nodes(self, file_path: str, repository: str) -> None:
        """Delete all nodes associated with a specific file in a repository."""

    @abstractmethod
    def get_repository_commit(self, repository: str) -> str | None:
        """Retrieve the last indexed commit hash for a repository."""

    @abstractmethod
    def set_repository_commit(self, repository: str, commit_hash: str) -> None:
        """Save the last indexed commit hash for a repository."""

    @abstractmethod
    def get_repository_metadata(self, repository: str) -> dict[str, Any] | None:
        """Retrieve repository metadata including last indexed commit, path, and last synced time."""
