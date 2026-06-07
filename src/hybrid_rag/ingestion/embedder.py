"""
Backward-compatibility shim.

The concrete Ollama embedder adapter now lives at:
  hybrid_rag.ingestion.ollama_embedder.OllamaEmbedder

This module re-exports it as Embedder so existing imports keep working.
New code should import from the port or the adapter directly.
"""

from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder as Embedder

__all__ = ["Embedder"]
