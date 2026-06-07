"""
Backward-compatibility shim.

The concrete FalkorDB adapter now lives at:
  hybrid_rag.graph.falkordb_store.FalkorDBStore

This module re-exports it as GraphClient so existing imports keep working.
New code should import from the port or the adapter directly.
"""

from hybrid_rag.graph.falkordb_store import FalkorDBStore as GraphClient

__all__ = ["GraphClient"]
