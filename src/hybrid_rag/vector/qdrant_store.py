"""
Qdrant adapter — implements VectorStore port.

Vendor: Qdrant vector database.
Swap this file for a different adapter (e.g. weaviate_store.py) to change backends.

Note: requires qdrant-client >= 1.12 (uses query_points, not deprecated .search).
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
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from hybrid_rag.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 6333
_DEFAULT_COLLECTION = "code_chunks"
_VECTOR_SIZE = 768  # nomic-embed-text output dimension


def _deterministic_uuid(node_id: str) -> str:
    """SHA-256 → stable UUID for idempotent upserts."""
    return str(uuid.UUID(bytes=hashlib.sha256(node_id.encode()).digest()[:16], version=4))


class QdrantStore(VectorStore):
    """VectorStore adapter backed by Qdrant."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
        vector_size: int = _VECTOR_SIZE,
        upsert_batch_size: int = 500,
    ) -> None:
        self._host = host or os.environ.get("QDRANT_HOST", _DEFAULT_HOST)
        self._port = int(port or os.environ.get("QDRANT_PORT", _DEFAULT_PORT))
        self._collection = collection or os.environ.get("QDRANT_COLLECTION", _DEFAULT_COLLECTION)
        self._vector_size = vector_size
        self._upsert_batch_size = upsert_batch_size
        self._client = QdrantClient(host=self._host, port=self._port)
        self._ensure_collection()
        logger.info(
            "QdrantStore connected: %s:%d collection=%s dim=%d",
            self._host,
            self._port,
            self._collection,
            self._vector_size,
        )

    # ── VectorStore interface ─────────────────────────────────────

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """
        Upsert a list of chunk dicts, automatically batched to stay under
        Qdrant's payload size limit.

        Each chunk must have: node_id, label, file_path, text, embedding.
        """
        points = [
            PointStruct(
                id=_deterministic_uuid(c["node_id"]),
                vector=c["embedding"],
                payload={
                    "node_id": c["node_id"],
                    "name": c.get("name", ""),
                    "label": c["label"],
                    "file_path": c["file_path"],
                    "text": c["text"],
                    "repository": c.get("repository", ""),
                    "file_type": c.get("file_type", ""),
                    "indexed_commit": c.get("indexed_commit", ""),
                    "index_run_id": c.get("index_run_id", ""),
                    "source_identity": c.get("source_identity", ""),
                },
            )
            for c in chunks
        ]
        if not points:
            return 0
        self._client.upload_points(
            collection_name=self._collection,
            points=points,
            batch_size=self._upsert_batch_size,
            wait=True,
        )
        logger.debug("Upserted %d points to %s", len(points), self._collection)
        return len(points)

    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top_k nearest chunks with their payloads and scores."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filter_payload:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_payload.items()
            ]
            query_filter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [{**hit.payload, "score": hit.score, "point_id": str(hit.id)} for hit in results]

    def point_count(self) -> int:
        res = self._client.count(collection_name=self._collection, exact=True)
        return res.count

    def clear(self) -> None:
        """Delete and recreate the collection. Use in tests only."""
        self._client.delete_collection(self._collection)
        self._ensure_collection()

    def delete_file_vectors(self, file_path: str, repository: str) -> None:
        """Delete all vectors associated with a specific file in a repository."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                    FieldCondition(key="repository", match=MatchValue(value=repository)),
                ]
            ),
        )

    def delete_repository(self, repository: str) -> None:
        """Delete one repository namespace without recreating the collection."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="repository", match=MatchValue(value=repository)),
                ]
            ),
            wait=True,
        )

    def delete_repository_except_run(self, repository: str, index_run_id: str) -> None:
        """Delete stale vectors after replacement data is present."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="repository", match=MatchValue(value=repository))],
                must_not=[FieldCondition(key="index_run_id", match=MatchValue(value=index_run_id))],
            ),
            wait=True,
        )

    def delete_file_vectors_except_run(
        self, file_path: str, repository: str, index_run_id: str
    ) -> None:
        """Delete stale vectors for one incrementally replaced file."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                    FieldCondition(key="repository", match=MatchValue(value=repository)),
                ],
                must_not=[FieldCondition(key="index_run_id", match=MatchValue(value=index_run_id))],
            ),
            wait=True,
        )

    def set_repository_metadata(self, repository: str, metadata: dict[str, Any]) -> None:
        """Stamp every repository vector after a successful indexing run."""
        self._client.set_payload(
            collection_name=self._collection,
            payload=metadata,
            points=Filter(
                must=[
                    FieldCondition(key="repository", match=MatchValue(value=repository)),
                ]
            ),
            wait=True,
        )

    def get_repository_metadata(self, repository: str) -> dict[str, set[Any]]:
        """Return distinct provenance values across all vectors in a repository."""
        fields = ("indexed_commit", "index_run_id", "source_identity")
        values: dict[str, set[Any]] = {field: set() for field in fields}
        repository_filter = Filter(
            must=[
                FieldCondition(key="repository", match=MatchValue(value=repository)),
            ]
        )
        for field in fields:
            response = self._client.facet(
                collection_name=self._collection,
                key=field,
                facet_filter=repository_filter,
                limit=10,
                exact=True,
            )
            values[field] = {hit.value for hit in response.hits if hit.value not in (None, "")}
        return values

    # ── Internal ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self._collection)
        for field_name in (
            "repository",
            "file_type",
            "indexed_commit",
            "index_run_id",
            "source_identity",
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                logger.warning("Unable to create Qdrant payload index for %s: %s", field_name, exc)
