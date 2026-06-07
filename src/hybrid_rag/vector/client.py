"""
Backward-compatibility shim.

The concrete Qdrant adapter now lives at:
  hybrid_rag.vector.qdrant_store.QdrantStore

This module re-exports it as VectorClient so existing imports keep working.
New code should import from the port or the adapter directly.
"""

from hybrid_rag.vector.qdrant_store import QdrantStore as VectorClient

__all__ = ["VectorClient"]
