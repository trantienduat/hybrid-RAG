"""
Port: BaseLLMExtractor — abstract interface for LLM-assisted KG extraction.

Concrete adapters:
  hybrid_rag.ingestion.ollama_llm_extractor.OllamaLLMExtractor

Any future LLM backend (OpenAI, Anthropic, vLLM…) must implement this
interface. Business logic depends only on BaseLLMExtractor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from hybrid_rag.ingestion.parser import EdgeData, ParseResult


class BaseLLMExtractor(ABC):
    """
    Contract for LLM-based supplemental KG extraction.

    Given the source text of a single file and its per-file ParseResult,
    returns additional EdgeData that static AST analysis cannot capture
    (e.g. USES edges from type annotations, typed attribute calls).
    """

    @abstractmethod
    def extract(self, source_text: str, result: ParseResult) -> list[EdgeData]:
        """
        Extract supplemental edges from source_text.

        Args:
            source_text: Raw source code of a single file.
            result:       ParseResult whose nodes serve as src_id candidates.
                          Typically a per-file result, but may be a repo-level
                          result — the implementation should filter as needed.

        Returns:
            List of EdgeData not already in result.edges.
            Returns [] on any error (never raises).
        """

    def close(self) -> None:
        """Release resources (HTTP connections, model handles, etc.)."""

    def __enter__(self) -> "BaseLLMExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
