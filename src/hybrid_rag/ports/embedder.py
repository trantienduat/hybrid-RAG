"""
Port: BaseEmbedder — abstract interface for text embedding.

Concrete adapters:
  hybrid_rag.ingestion.ollama_embedder.OllamaEmbedder

Any future embedding backend (OpenAI, HuggingFace, Cohere…) must implement
this interface. Business logic depends only on BaseEmbedder, never on a vendor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hybrid_rag.ingestion.parser import NodeData


class BaseEmbedder(ABC):
    """Contract for embedding backends."""

    @abstractmethod
    def embed_nodes(self, nodes: list[NodeData]) -> list[dict[str, Any]]:
        """
        Embed each node.

        Returns a list of chunk dicts (node_id, label, file_path, text, embedding)
        ready for VectorStore.upsert().
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a free-text query string for similarity search."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple text strings, potentially concurrently.

        Returns a list of embeddings in the same order as the input texts.
        """
        return [self.embed_query(t) for t in texts]

    def close(self) -> None:
        """Release any underlying HTTP connections or resources."""

    def __enter__(self) -> BaseEmbedder:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
