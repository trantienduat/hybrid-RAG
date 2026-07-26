"""
Vector Retriever — semantic search over indexed code chunks via Qdrant.

Embeds the query with the configured embedder, then runs cosine-similarity
search on the vector store.  Thin wrapper over the VectorStore + BaseEmbedder
ports — fully vendor-neutral.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Any

from hybrid_rag.ports.embedder import BaseEmbedder
from hybrid_rag.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 10


def _base_node_id(node_id: str) -> str:
    """Strip trailing ``::N`` chunk-index suffix to get the canonical node ID.

    Examples::
        "math_utils.py::MyClass::add::0"  -> "math_utils.py::MyClass::add"
        "math_utils.py::MyClass::0"       -> "math_utils.py::MyClass"
        "math_utils.py::add::0"           -> "math_utils.py::add"
        "plain_id"                         -> "plain_id"
    """
    parts = node_id.rsplit("::", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return node_id


class VectorRetriever:
    """Retrieve relevant code chunks via embedding similarity."""

    def __init__(self, vector_store: VectorStore, embedder: BaseEmbedder) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = Lock()

    def retrieve(
        self,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Embed *query* and return top_k nearest code chunks.

        Each result dict has:
          node_id, base_node_id, label, file_path, text, score, source.
        """
        with self._cache_lock:
            embedding = self._embedding_cache.get(query)
        if embedding is None:
            embedding = self._embedder.embed_query(query)
            with self._cache_lock:
                self._embedding_cache[query] = embedding
                if len(self._embedding_cache) > 256:
                    self._embedding_cache.popitem(last=False)
        else:
            with self._cache_lock:
                self._embedding_cache.move_to_end(query)
        hits = self._store.search(embedding, top_k=top_k, filter_payload=filter_payload)
        results: list[dict[str, Any]] = []
        for hit in hits:
            nid = hit.get("node_id", "")
            results.append(
                {
                    **hit,
                    "base_node_id": _base_node_id(nid),
                    "source": "vector",
                }
            )
        logger.debug("VectorRetriever: %d hits for %r", len(results), query[:60])
        return results

    def clear_cache(self) -> None:
        """Clear cached query embeddings."""
        with self._cache_lock:
            self._embedding_cache.clear()
