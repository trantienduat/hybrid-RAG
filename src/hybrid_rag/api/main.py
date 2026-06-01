"""
FastAPI application — hybrid-rag REST API.

M4 #27 (endpoints) + #29 (streaming SSE).

Endpoints:
  GET  /health                     — service health check
  POST /query                      — hybrid retrieval + LLM answer (sync)
  POST /query/stream               — hybrid retrieval + LLM answer (SSE)
  GET  /graph/neighbors/{node_id}  — direct neighbors of a KG node (for D3)
  GET  /graph/search               — find nodes by name substring
  GET  /                           — D3.js graph explorer frontend
  GET  /static/*                   — static assets

Start with:
  hybrid-rag serve
  # or:
  uvicorn hybrid_rag.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from hybrid_rag.api.schemas import (
    GraphNeighborsResponse,
    GraphNode,
    GraphSearchResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.retrieval.query_analyzer import analyze
from hybrid_rag.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# ── Configuration from environment ────────────────────────────────────────────

_FALKORDB_HOST       = os.environ.get("FALKORDB_HOST") or "localhost"
_FALKORDB_PORT       = int(os.environ.get("FALKORDB_PORT") or 6379)
_FALKORDB_GRAPH      = os.environ.get("FALKORDB_GRAPH") or "codebase"
_QDRANT_HOST         = os.environ.get("QDRANT_HOST") or "localhost"
_QDRANT_PORT         = int(os.environ.get("QDRANT_PORT") or 6333)
_QDRANT_COLLECTION   = os.environ.get("QDRANT_COLLECTION") or "code_chunks"
_OLLAMA_URL          = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
_EMBED_MODEL         = os.environ.get("EMBED_MODEL") or "nomic-embed-text"
_RRF_K               = int(os.environ.get("RRF_K", 60))
_RRF_STRUCTURAL_W    = float(os.environ.get("RRF_STRUCTURAL_WEIGHT", 3.0))
_RRF_HYBRID_W        = float(os.environ.get("RRF_HYBRID_WEIGHT", 1.5))


# ── Lifespan: wire up stores once ──────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("hybrid-rag API starting — initialising stores")
    app.state.graph_store = FalkorDBStore(
        host=_FALKORDB_HOST, port=_FALKORDB_PORT, graph_name=_FALKORDB_GRAPH
    )
    app.state.vector_store = QdrantStore(
        host=_QDRANT_HOST, port=_QDRANT_PORT, collection=_QDRANT_COLLECTION
    )
    app.state.embedder = OllamaEmbedder(ollama_url=_OLLAMA_URL, model=_EMBED_MODEL)
    app.state.retriever = HybridRetriever(
        graph_store=app.state.graph_store,
        vector_store=app.state.vector_store,
        embedder=app.state.embedder,
        rrf_k=_RRF_K,
        rrf_structural_weight=_RRF_STRUCTURAL_W,
        rrf_hybrid_weight=_RRF_HYBRID_W,
    )
    logger.info("hybrid-rag API ready")
    yield
    app.state.embedder.close()
    logger.info("hybrid-rag API shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="hybrid-rag",
    description="Privacy-preserving Graph-Hybrid RAG for relationship-aware codebase understanding.",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_source_chunks(results: list[dict[str, Any]], n: int) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for r in results[:n]:
        chunks.append(SourceChunk(
            node_id=r.get("node_id", ""),
            name=r.get("name", ""),
            label=r.get("label", ""),
            file_path=r.get("file_path", ""),
            text=r.get("text", ""),
            source=r.get("source", ""),
            rrf_score=round(float(r.get("rrf_score", 0.0)), 6),
        ))
    return chunks


def _build_prompt(question: str, context: str, is_global: bool = False) -> str:
    if is_global:
        return (
            "You are an expert principal software architect. Below is a set of hierarchical community summaries "
            "describing the structural design, modules, and dependencies of the codebase.\n"
            "Analyze these summaries and provide a comprehensive, highly-structured architectural report. "
            "Highlight key components, database models, core flows, and cross-module relationships.\n\n"
            f"Community Summaries Context:\n{context}\n\n"
            f"User Request: {question}\n\n"
            "Architectural Report:"
        )
    return (
        "You are an expert code assistant. Use ONLY the context below to answer the question. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def _ollama_generate(prompt: str, model: str) -> str:
    """Call Ollama /api/generate (non-streaming)."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{_OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


async def _ollama_stream(prompt: str, model: str) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from Ollama /api/generate (streaming)."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{_OLLAMA_URL}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                done = bool(data.get("done", False))
                yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                if done:
                    break


# ── GET / ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the D3.js graph explorer frontend."""
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index)


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check reachability of FalkorDB, Qdrant, and Ollama."""
    results: dict[str, str] = {}
    overall = "ok"

    try:
        n = app.state.graph_store.node_count()
        results["falkordb"] = f"ok ({n} nodes)"
    except Exception as exc:  # noqa: BLE001
        results["falkordb"] = f"error: {exc}"
        overall = "degraded"

    try:
        n = app.state.vector_store.point_count()
        results["qdrant"] = f"ok ({n} points)"
    except Exception as exc:  # noqa: BLE001
        results["qdrant"] = f"error: {exc}"
        overall = "degraded"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{_OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
        results["ollama"] = f"ok ({len(models)} models)"
    except Exception as exc:  # noqa: BLE001
        results["ollama"] = f"error: {exc}"
        overall = "degraded"

    return HealthResponse(status=overall, **results)


# ── POST /query ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Hybrid retrieval + LLM answer (synchronous)."""
    t0 = time.perf_counter()

    analysis = analyze(req.question)
    retriever: HybridRetriever = app.state.retriever

    try:
        # Route budget parameters
        if req.max_tokens is None and req.max_chars is None:
            ctx = retriever.retrieve_with_context(
                req.question, top_k=req.top_k, max_tokens=None, max_chars=None, context_n=req.context_n
            )
        else:
            ctx = retriever.retrieve_with_context(
                req.question, top_k=req.top_k, max_tokens=req.max_tokens, max_chars=req.max_chars
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc

    prompt = _build_prompt(req.question, ctx.text or "(no code context retrieved)", is_global=(analysis.query_type == "global"))

    try:
        answer = await _ollama_generate(prompt, req.llm_model)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    # Build response sources from actually packed context chunks
    sources = [
        SourceChunk(
            node_id=r.get("node_id", ""),
            name=r.get("name", ""),
            label=r.get("label", ""),
            file_path=r.get("file_path", ""),
            text=r.get("text", ""),
            source=r.get("source", ""),
            rrf_score=round(float(r.get("rrf_score", 0.0)), 6),
        )
        for r in ctx.chunks
    ]

    return QueryResponse(
        question=req.question,
        answer=answer,
        query_type=analysis.query_type,
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


# ── POST /query/stream ─────────────────────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    Hybrid retrieval + LLM answer — Server-Sent Events stream.

    First event: {"token": "", "done": false, "sources": [...], "query_type": "..."}
    Subsequent: {"token": "<word>", "done": false}
    Final:       {"token": "", "done": true}
    """
    analysis = analyze(req.question)
    retriever: HybridRetriever = app.state.retriever

    try:
        # Route budget parameters
        if req.max_tokens is None and req.max_chars is None:
            ctx = retriever.retrieve_with_context(
                req.question, top_k=req.top_k, max_tokens=None, max_chars=None, context_n=req.context_n
            )
        else:
            ctx = retriever.retrieve_with_context(
                req.question, top_k=req.top_k, max_tokens=req.max_tokens, max_chars=req.max_chars
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc

    prompt = _build_prompt(req.question, ctx.text or "(no code context retrieved)", is_global=(analysis.query_type == "global"))
    
    sources_payload = [
        {
            "node_id": r.get("node_id", ""),
            "name": r.get("name", ""),
            "label": r.get("label", ""),
            "file_path": r.get("file_path", ""),
            "text": r.get("text", ""),
            "source": r.get("source", ""),
            "rrf_score": round(float(r.get("rrf_score", 0.0)), 6),
        }
        for r in ctx.chunks
    ]

    async def _event_stream() -> AsyncIterator[str]:
        # First event: metadata
        meta = json.dumps({
            "token": "",
            "done": False,
            "sources": sources_payload,
            "query_type": analysis.query_type,
        })
        yield f"data: {meta}\n\n"

        # Stream LLM tokens
        async for chunk in _ollama_stream(prompt, req.llm_model):
            yield chunk

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /graph/neighbors/{node_id} ─────────────────────────────────────────────

@app.get("/graph/neighbors/{node_id:path}", response_model=GraphNeighborsResponse)
async def graph_neighbors(
    node_id: str,
    direction: str = Query("both", pattern="^(in|out|both)$"),
    limit: int = Query(30, ge=1, le=100),
) -> GraphNeighborsResponse:
    """Return direct neighbors of *node_id* for graph exploration."""
    store: FalkorDBStore = app.state.graph_store

    # Resolve the anchor node metadata
    simple_name = node_id.rsplit("::", 1)[-1]
    if "." in simple_name:
        simple_name = simple_name.rsplit(".", 1)[-1]
    
    nodes = store.find_nodes(simple_name, limit=5)
    anchor = next((n for n in nodes if n.get("node_id") == node_id), None)
    label = anchor.get("label", "") if anchor else ""
    name = anchor.get("name", simple_name) if anchor else node_id

    directions = ["in", "out"] if direction == "both" else [direction]
    seen: set[str] = set()
    edges = []
    for d in directions:
        try:
            raw = store.find_neighbors(node_id, direction=d, max_hops=1, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("find_neighbors failed for %s direction=%s: %s", node_id, d, exc)
            raw = []
        for nb in raw:
            key = (nb.get("rel", ""), nb.get("dst_id", ""))
            if key not in seen:
                seen.add(key)
                edges.append({
                    "src_id": node_id,
                    "rel": nb.get("rel", ""),
                    "dst_id": nb.get("dst_id", ""),
                    "dst_name": nb.get("dst_name", ""),
                    "dst_label": nb.get("dst_label", ""),
                })

    return GraphNeighborsResponse(
        node_id=node_id, label=label, name=name, neighbors=edges
    )


# ── GET /graph/search ──────────────────────────────────────────────────────────

@app.get("/graph/search", response_model=GraphSearchResponse)
async def graph_search(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=50),
) -> GraphSearchResponse:
    """Find KG nodes whose name contains *q*."""
    store: FalkorDBStore = app.state.graph_store
    try:
        raw = store.find_nodes(q, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    nodes = [
        GraphNode(
            id=n.get("node_id", ""),
            name=n.get("name", ""),
            label=n.get("label", ""),
            file_path=n.get("file_path", ""),
        )
        for n in raw
    ]
    return GraphSearchResponse(query=q, nodes=nodes)
