"""
Port: BaseRetriever — abstract interface for retrieval pipelines.

Concrete implementations:
  hybrid_rag.retrieval.hybrid_retriever.HybridRetriever

Any future retrieval strategy (BM25+KG, dense+sparse, …) must implement
this interface. The CLI and API depend only on BaseRetriever, never on a
specific retrieval strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Contract for retrieval backends."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve top_k relevant context items for a query.

        Each result dict contains at least:
          node_id    (str)   — canonical node identifier
          text       (str)   — code/text chunk (may be empty for graph-only results)
          rrf_score  (float) — fusion score (higher = more relevant)
          source     (str)   — "graph" | "vector" | "hybrid"
        """

    def close(self) -> None:
        """Release any underlying connections or resources."""

    def __enter__(self) -> BaseRetriever:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
