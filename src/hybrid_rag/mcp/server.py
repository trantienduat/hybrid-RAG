"""
Model Context Protocol (MCP) server implementation for Hybrid-RAG.

Exposes tools to query codebase, find entities, and trace relationships.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hybrid_rag.config import app_config, translate_path_for_docker
from hybrid_rag.eval.preflight import EvaluationPreflightError, validate_index_provenance
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.vector.qdrant_store import QdrantStore

logger = logging.getLogger("hybrid_rag.mcp")

# Initialize FastMCP Server
mcp = FastMCP("hybrid-rag")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Check MCP dependencies, not only the HTTP process."""
    dependencies: dict[str, Any] = {}
    try:
        graph_store, vector_store, _ = get_components()
        dependencies["falkordb"] = {"status": "ok", "nodes": graph_store.node_count()}
        dependencies["qdrant"] = {"status": "ok", "points": vector_store.point_count()}
    except Exception as exc:
        logger.exception("MCP storage health check failed")
        dependencies["storage"] = {"status": "error", "detail": str(exc)}
        return JSONResponse(
            {"status": "degraded", "dependencies": dependencies},
            status_code=503,
        )

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{_OLLAMA_URL.rstrip('/')}/api/tags")
            response.raise_for_status()
        dependencies["ollama"] = {"status": "ok"}
    except Exception as exc:
        logger.exception("MCP Ollama health check failed")
        dependencies["ollama"] = {"status": "error", "detail": str(exc)}
        return JSONResponse(
            {"status": "degraded", "dependencies": dependencies},
            status_code=503,
        )

    return JSONResponse({"status": "ok", "dependencies": dependencies})


# ── Configuration from environment/config file ──────────────────────────────────

_FALKORDB_HOST = app_config.falkordb_host
_FALKORDB_PORT = app_config.falkordb_port
_FALKORDB_GRAPH = app_config.falkordb_graph
_QDRANT_HOST = app_config.qdrant_host
_QDRANT_PORT = app_config.qdrant_port
_QDRANT_COLLECTION = app_config.qdrant_collection
_OLLAMA_URL = app_config.ollama_url
_EMBED_MODEL = app_config.embed_model
_RRF_K = app_config.rrf_k
_RRF_STRUCTURAL_W = app_config.rrf_structural_weight
_RRF_HYBRID_W = app_config.rrf_hybrid_weight


# Global caches for lazy initialization
_graph_store: FalkorDBStore | None = None
_vector_store: QdrantStore | None = None
_retriever: HybridRetriever | None = None
_embedder: OllamaEmbedder | None = None


def _index_status(
    graph_store: FalkorDBStore,
    vector_store: QdrantStore,
    repository: str,
) -> dict[str, Any]:
    metadata = graph_store.get_repository_metadata(repository) or {}
    indexed_commit = metadata.get("last_indexed_commit")
    current_commit = None
    working_tree_dirty = None
    provenance_valid = True
    provenance_error = None

    try:
        validate_index_provenance(graph_store, vector_store, repository)
    except EvaluationPreflightError as exc:
        provenance_valid = False
        provenance_error = str(exc)

    configured_path = app_config.get_repo_path(repository) or metadata.get("source_path")
    if configured_path:
        repo_path = Path(translate_path_for_docker(configured_path))
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            current_commit = result.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            working_tree_dirty = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            logger.debug("Unable to determine current commit for %s", repository)

    if not provenance_valid:
        stale = True
    elif working_tree_dirty:
        stale = True
    elif indexed_commit is not None and current_commit is not None:
        stale = indexed_commit != current_commit
    else:
        stale = None
    return {
        "repository": repository,
        "indexed_commit": indexed_commit,
        "current_commit": current_commit,
        "working_tree_dirty": working_tree_dirty,
        "provenance_valid": provenance_valid,
        "provenance_error": provenance_error,
        "stale": stale,
        "updated_at": metadata.get("updated_at"),
    }


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
    max_tokens: int = 1024,
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
        graph_store, vector_store, retriever = get_components()
        ctx = retriever.retrieve_with_context(
            query=question,
            top_k=top_k,
            max_tokens=max_tokens,
            repository=repository,
        )
        text = ctx.text or "(No relevant codebase context was retrieved for this query.)"
        if repository:
            status = _index_status(graph_store, vector_store, repository)
            if status["stale"]:
                if status.get("provenance_valid", True) is False:
                    reason = status["provenance_error"]
                elif status.get("working_tree_dirty"):
                    reason = "the configured working tree has uncommitted changes"
                else:
                    reason = (
                        f"indexed {status['indexed_commit']}, current {status['current_commit']}"
                    )
                text = f"[Index warning: {repository} is stale; {reason}]\n\n{text}"
        return text
    except Exception as exc:
        logger.exception("MCP query_codebase failed")
        raise ToolError("Codebase retrieval failed; check MCP dependency health.") from exc


@mcp.tool()
def list_repositories() -> list[str]:
    """List all unique repository names indexed in the database."""
    try:
        graph_store, _, _ = get_components()
        return graph_store.list_repositories()
    except Exception:
        logger.exception("MCP list_repositories failed")
        raise ToolError("Unable to list indexed repositories.") from None


@mcp.tool()
def get_index_status(repository: str) -> dict[str, Any]:
    """Return indexed/current commits and whether a repository index is stale."""
    try:
        graph_store, vector_store, _ = get_components()
        return _index_status(graph_store, vector_store, repository)
    except Exception as exc:
        logger.exception("MCP get_index_status failed")
        raise ToolError(f"Unable to inspect index status for {repository}.") from exc


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
        raise ToolError("AST node search failed.") from None


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
        raise ToolError(f"Neighbor traversal failed for {node_id}.") from None


@mcp.tool()
def get_community_report(repository: str | None = None) -> list[dict[str, Any]]:
    """Retrieve Directory-based community partitioning summaries for architectural queries.

    Args:
        repository: Scope community summaries to a specific repository namespace (optional).
    """
    try:
        graph_store, _, _ = get_components()
        if repository:
            cypher = (
                "MATCH (c:Community)<-[:IN_COMMUNITY]-(n) "
                "WHERE n.repository = $repository "
                "RETURN DISTINCT c.id AS id, c.name AS name, "
                "c.summary AS summary, c.level AS level"
            )
            res = graph_store.query(cypher, {"repository": repository})
        else:
            cypher = (
                "MATCH (c:Community) "
                "RETURN c.id AS id, c.name AS name, c.summary AS summary, c.level AS level"
            )
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
        raise ToolError("Unable to retrieve community reports.") from None
