#!/usr/bin/env python3
"""
Reset script for Hybrid-RAG Live Demo.
Completely purges 'demo-checkout' from FalkorDB and Qdrant, and cleans up any added extension files.
Ensures full reproducibility before starting the demo.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hybrid_rag.config import app_config
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.vector.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reset_demo")

REPO_NAME = "demo-checkout"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "demo_checkout"


def reset_demo() -> None:
    logger.info("=== Resetting Demo Checkout Fixture & Stores ===")

    # 1. Clean up local extension file
    ext_file = FIXTURE_DIR / "logger.py"
    if ext_file.exists():
        ext_file.unlink()
        logger.info("✓ Removed extension file: %s", ext_file)
    else:
        logger.info("✓ No leftover %s found", ext_file.name)

    # 2. Delete from FalkorDB
    try:
        graph_store = FalkorDBStore(
            host=app_config.falkordb_host,
            port=app_config.falkordb_port,
            graph_name=app_config.falkordb_graph,
        )
        graph_store.delete_repository(REPO_NAME)
        logger.info("✓ FalkorDB: Purged all graph nodes & edges for repository '%s'", REPO_NAME)
    except Exception as exc:
        logger.warning("FalkorDB purge warning: %s", exc)

    # 3. Delete from Qdrant
    try:
        vector_store = QdrantStore(
            host=app_config.qdrant_host,
            port=app_config.qdrant_port,
            collection=app_config.qdrant_collection,
        )
        vector_store.delete_repository(REPO_NAME)
        logger.info("✓ Qdrant: Purged all vector chunks for repository '%s'", REPO_NAME)
    except Exception as exc:
        logger.warning("Qdrant purge warning: %s", exc)

    logger.info("🎉 Clean slate ready! You can now start the Live Demo from Act I.")


if __name__ == "__main__":
    reset_demo()
