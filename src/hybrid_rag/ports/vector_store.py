"""
Port: VectorStore — abstract interface for vector storage and similarity search.

Concrete adapters:
  hybrid_rag.vector.qdrant_store.QdrantStore

Any future vector backend (Weaviate, Pinecone, pgvector…) must implement
this interface. Business logic depends only on VectorStore, never on a vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Contract for vector store backends."""

    @abstractmethod
    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """
        Upsert a list of chunk dicts.

        Each chunk must have: node_id, label, file_path, text, embedding.
        Returns the number of points upserted.
        """

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return top_k nearest chunks by cosine similarity.

        Each result dict contains at least: node_id, label, file_path, text, score.
        """

    @abstractmethod
    def point_count(self) -> int:
        """Return total number of stored vectors."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all vectors. Intended for tests only."""

    @abstractmethod
    def delete_file_vectors(self, file_path: str, repository: str) -> None:
        """Delete all vectors associated with a specific file in a repository."""

    @abstractmethod
    def delete_repository(self, repository: str) -> None:
        """Delete vectors for one repository namespace."""

    @abstractmethod
    def set_repository_metadata(self, repository: str, metadata: dict[str, Any]) -> None:
        """Stamp all vectors for a repository with the current index provenance."""

    @abstractmethod
    def get_repository_metadata(self, repository: str) -> dict[str, set[Any]]:
        """Return distinct provenance values stored on a repository's vectors."""
