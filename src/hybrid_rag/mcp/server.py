"""
Model Context Protocol (MCP) server implementation for Hybrid-RAG.

Exposes tools to query codebase, find entities, and trace relationships.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.vector.qdrant_store import QdrantStore
from hybrid_rag.constants import DEFAULT_EMBED_MODEL

logger = logging.getLogger("hybrid_rag.mcp")

# Initialize FastMCP Server
mcp = FastMCP("hybrid-rag")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Check health of the MCP server."""
    return JSONResponse({"status": "ok"})


# ── Configuration from environment (aligned with REST API) ────────────────────

_FALKORDB_HOST = os.environ.get("FALKORDB_HOST") or "localhost"
_FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT") or 6379)
_FALKORDB_GRAPH = os.environ.get("FALKORDB_GRAPH") or "codebase"
_QDRANT_HOST = os.environ.get("QDRANT_HOST") or "localhost"
_QDRANT_PORT = int(os.environ.get("QDRANT_PORT") or 6333)
_QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION") or "code_chunks"
_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
_EMBED_MODEL = os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL
_RRF_K = int(os.environ.get("RRF_K", 60))
_RRF_STRUCTURAL_W = float(os.environ.get("RRF_STRUCTURAL_WEIGHT", 3.0))
_RRF_HYBRID_W = float(os.environ.get("RRF_HYBRID_WEIGHT", 1.5))

# Global caches for lazy initialization
_graph_store: FalkorDBStore | None = None
_vector_store: QdrantStore | None = None
_retriever: HybridRetriever | None = None
_embedder: OllamaEmbedder | None = None


def get_components() -> tuple[FalkorDBStore, QdrantStore, HybridRetriever]:
    """Lazily initialize and return the store and retriever instances."""
    global _retriever, _graph_store, _vector_store, _embedder
    if _retriever is None:
        logger.info("Initializing DB connections and HybridRetriever for MCP...")
        _graph_store = FalkorDBStore(
            host=_FALKORDB_HOST, port=_FALKORDB_PORT, graph_name=_FALKORDB_GRAPH
        )
        _vector_store = QdrantStore(
            host=_QDRANT_HOST, port=_QDRANT_PORT, collection=_QDRANT_COLLECTION
        )
        _embedder = OllamaEmbedder(ollama_url=_OLLAMA_URL, model=_EMBED_MODEL)
        _retriever = HybridRetriever(
            graph_store=_graph_store,
            vector_store=_vector_store,
            embedder=_embedder,
            rrf_k=_RRF_K,
            rrf_structural_weight=_RRF_STRUCTURAL_W,
            rrf_hybrid_weight=_RRF_HYBRID_W,
        )
    return _graph_store, _vector_store, _retriever


# ── MCP Tools ──────────────────────────────────────────────────────────────────


@mcp.tool()
def query_codebase(
    question: str,
    repository: str | None = None,
    top_k: int = 20,
    max_tokens: int = 2048,
) -> str:
    """Query the indexed codebase using hybrid graph + vector retrieval with RRF.

    Use this tool to find relevant code snippets, functions, classes, and their context
    to answer codebase questions.

    Args:
        question: The natural language query or question about the codebase.
        repository: Scope search to a specific repository name (optional).
        top_k: Number of candidate chunks to retrieve initially (default: 20).
        max_tokens: Limit on assembled context token size (default: 2048).
    """
    try:
        _, _, retriever = get_components()
        ctx = retriever.retrieve_with_context(
            query=question,
            top_k=top_k,
            max_tokens=max_tokens,
            repository=repository,
        )
        return ctx.text or "(No relevant codebase context was retrieved for this query.)"
    except Exception as exc:
        logger.exception("MCP query_codebase failed")
        return f"Error executing retrieval: {exc}"


@mcp.tool()
def list_repositories() -> list[str]:
    """List all unique repository names indexed in the database."""
    try:
        graph_store, _, _ = get_components()
        return graph_store.list_repositories()
    except Exception:
        logger.exception("MCP list_repositories failed")
        return []


@mcp.tool()
def search_ast_nodes(
    query: str,
    repository: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search for code entities (modules, classes, methods, functions) by name.

    Args:
        query: Part or all of the entity name to search for (case-sensitive).
        repository: Scope query to a specific repository namespace (optional).
        limit: Max nodes to return (default: 20, max: 50).
    """
    try:
        graph_store, _, _ = get_components()
        sanitized_limit = min(max(1, limit), 50)
        return graph_store.find_nodes(name=query, repository=repository, limit=sanitized_limit)
    except Exception:
        logger.exception("MCP search_ast_nodes failed")
        return []


@mcp.tool()
def get_ast_neighbors(
    node_id: str,
    direction: str = "both",
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Traverse incoming/outgoing AST relationships for a specific entity.

    Exposes class inheritance, function calls, imports, and module definition structures.

    Args:
        node_id: Fully Qualified Name (FQN) of the node (e.g. package.module.Class.method).
        direction: Direction to traverse: 'in' (incoming), 'out' (outgoing), or 'both' (default).
        limit: Max relationship records to return (default: 30).
    """
    if direction not in ("in", "out", "both"):
        direction = "both"

    try:
        graph_store, _, _ = get_components()
        directions = ["in", "out"] if direction == "both" else [direction]
        edges = []
        seen = set()
        for d in directions:
            raw = graph_store.find_neighbors(node_id=node_id, direction=d, max_hops=1, limit=limit)
            for nb in raw:
                # Deduplicate edges returned from both directions
                key = (nb.get("rel", ""), nb.get("dst_id", ""))
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        {
                            "src_id": node_id,
                            "rel": nb.get("rel", ""),
                            "dst_id": nb.get("dst_id", ""),
                            "dst_name": nb.get("dst_name", ""),
                            "dst_label": nb.get("dst_label", ""),
                            "dst_file_path": nb.get("dst_file_path", ""),
                            "dst_repository": nb.get("dst_repository", ""),
                        }
                    )
        return edges
    except Exception:
        logger.exception("MCP get_ast_neighbors failed")
        return []


@mcp.tool()
def get_community_report(repository: str | None = None) -> list[dict[str, Any]]:
    """Retrieve Louvain community partitioning summaries for architectural queries.

    Args:
        repository: Scope community summaries to a specific repository namespace (optional).
    """
    try:
        graph_store, _, _ = get_components()
        # Louvain communities are stored with 'Community' label in FalkorDB
        cypher = "MATCH (c:Community) RETURN c.id AS id, c.name AS name, c.summary AS summary, c.level AS level"
        res = graph_store.query(cypher)
        communities = []
        for row in res.result_set or []:
            communities.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "summary": row[2],
                    "level": row[3] if len(row) > 3 else 0,
                }
            )
        return communities
    except Exception:
        logger.exception("MCP get_community_report failed")
        return []
