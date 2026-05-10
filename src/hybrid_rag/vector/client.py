"""
Qdrant vector client: upsert code chunk embeddings.

Collection name: configured via QDRANT_COLLECTION env var (default: "code_chunks").
Points are identified by a deterministic UUID derived from the chunk's source id.

Payload schema per point:
  - node_id: str   (maps back to KG node id)
  - label: str     (Module | Class | Function | Variable)
  - file_path: str
  - text: str      (the chunk text that was embedded)
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 6333
_DEFAULT_COLLECTION = "code_chunks"
_VECTOR_SIZE = 768  # nomic-embed-text output dimension


def _deterministic_uuid(node_id: str) -> str:
    """SHA-256 → UUID5 for stable point IDs."""
    return str(uuid.UUID(bytes=hashlib.sha256(node_id.encode()).digest()[:16], version=4))


class VectorClient:
    """Thin wrapper around Qdrant for code chunk upsert/search."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
        vector_size: int = _VECTOR_SIZE,
    ) -> None:
        self._host = host or os.environ.get("QDRANT_HOST", _DEFAULT_HOST)
        self._port = int(port or os.environ.get("QDRANT_PORT", _DEFAULT_PORT))
        self._collection = collection or os.environ.get("QDRANT_COLLECTION", _DEFAULT_COLLECTION)
        self._vector_size = vector_size
        self._client = QdrantClient(host=self._host, port=self._port)
        self._ensure_collection()
        logger.info(
            "VectorClient connected: %s:%d collection=%s dim=%d",
            self._host, self._port, self._collection, self._vector_size,
        )

    # ── Public API ────────────────────────────────────────────────

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """
        Upsert a list of chunk dicts.

        Each chunk must have:
          - node_id: str
          - label: str
          - file_path: str
          - text: str
          - embedding: list[float]
        """
        points = [
            PointStruct(
                id=_deterministic_uuid(c["node_id"]),
                vector=c["embedding"],
                payload={
                    "node_id": c["node_id"],
                    "label": c["label"],
                    "file_path": c["file_path"],
                    "text": c["text"],
                },
            )
            for c in chunks
        ]
        if not points:
            return 0
        self._client.upsert(collection_name=self._collection, points=points)
        logger.debug("Upserted %d points to %s", len(points), self._collection)
        return len(points)

    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top_k nearest chunks with their payloads and scores."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filter_payload:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_payload.items()
            ]
            query_filter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [
            {**hit.payload, "score": hit.score, "point_id": str(hit.id)}
            for hit in results
        ]

    def point_count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def clear(self) -> None:
        """Delete and recreate the collection. Use in tests only."""
        self._client.delete_collection(self._collection)
        self._ensure_collection()

    # ── Internal ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self._collection)
