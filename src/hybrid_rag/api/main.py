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

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from hybrid_rag.api.schemas import (
    GraphNeighborsResponse,
    GraphNode,
    GraphSearchResponse,
    HealthResponse,
    IndexRequest,
    IndexTaskDetailResponse,
    IndexTaskResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.retrieval.query_analyzer import analyze
from hybrid_rag.utils.tracing import initialize_tracing, start_span
from hybrid_rag.vector.qdrant_store import QdrantStore

from hybrid_rag.utils.cache import RedisQueryCache
from hybrid_rag.constants import DEFAULT_LLM_MODEL, DEFAULT_EMBED_MODEL
import gc

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _is_gemini_provider(model: str, provider_env_var: str | None = None) -> bool:
    """Check if the given model or environment override indicates Gemini provider."""
    if model.startswith("gemini") or model == "text-embedding-004":
        return True
    if provider_env_var and os.environ.get(provider_env_var) == "gemini":
        return True
    return False


# ── Configuration from environment ────────────────────────────────────────────

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


# ── Lifespan: wire up stores once ──────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("hybrid-rag API starting — initialising stores")
    # Initialize OpenTelemetry Tracing for LLM Observability
    initialize_tracing()

    app.state.graph_store = FalkorDBStore(
        host=_FALKORDB_HOST, port=_FALKORDB_PORT, graph_name=_FALKORDB_GRAPH
    )
    app.state.vector_store = QdrantStore(
        host=_QDRANT_HOST, port=_QDRANT_PORT, collection=_QDRANT_COLLECTION
    )

    if _is_gemini_provider(_EMBED_MODEL, "EMBED_PROVIDER"):
        from hybrid_rag.ingestion.gemini_embedder import GeminiEmbedder

        app.state.embedder = GeminiEmbedder(model=_EMBED_MODEL)
        logger.info("Initialized GeminiEmbedder with model %s", _EMBED_MODEL)
    else:
        app.state.embedder = OllamaEmbedder(ollama_url=_OLLAMA_URL, model=_EMBED_MODEL)
        logger.info("Initialized OllamaEmbedder with model %s", _EMBED_MODEL)

    app.state.retriever = HybridRetriever(
        graph_store=app.state.graph_store,
        vector_store=app.state.vector_store,
        embedder=app.state.embedder,
        rrf_k=_RRF_K,
        rrf_structural_weight=_RRF_STRUCTURAL_W,
        rrf_hybrid_weight=_RRF_HYBRID_W,
    )
    app.state.indexing_tasks = {}
    app.state.indexing_lock = asyncio.Lock()
    
    # Initialize Concurrency Guard and Redis Query Cache
    app.state.llm_semaphore = asyncio.Semaphore(1)
    app.state.query_cache = RedisQueryCache()
    await app.state.query_cache.connect()
    
    logger.info("hybrid-rag API ready")
    yield
    app.state.embedder.close()
    await app.state.query_cache.disconnect()
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
        chunks.append(
            SourceChunk(
                node_id=r.get("node_id", ""),
                name=r.get("name", ""),
                label=r.get("label", ""),
                file_path=r.get("file_path", ""),
                text=r.get("text", ""),
                source=r.get("source", ""),
                rrf_score=round(float(r.get("rrf_score", 0.0)), 6),
            )
        )
    return chunks


def _build_prompt(question: str, context: str, is_global: bool = False) -> str:
    if is_global:
        return (
            "You are an expert principal software architect. Below is a set of hierarchical community summaries "
            "describing the structural design, modules, and dependencies of the codebase.\n"
            "Analyze these summaries and provide a comprehensive, highly-structured architectural report. "
            "Highlight key components, database models, core flows, and cross-module relationships.\n"
            "If the summaries are sparse, combine these structural clues with your general software architecture knowledge "
            "to infer design patterns, architectures, and intent.\n\n"
            "CRITICAL: You MUST write your detailed, step-by-step reasoning process inside <think> and </think> tags FIRST, "
            "and then write your final report outside the tags. You must strictly follow this format:\n"
            "<think>\n"
            "[Your detailed code analysis, module reviews, and thinking steps]\n"
            "</think>\n\n"
            "[Your final architectural report]\n\n"
            f"Community Summaries Context:\n{context}\n\n"
            f"User Request: {question}\n\n"
            "Architectural Report:"
        )
    return (
        "You are an expert code assistant. Use the provided context below as the primary source of truth to answer the question.\n"
        "If the context is sparse (e.g. only contains a list of directories, modules, or file definitions) but lacks conceptual detail, "
        "you should synthesize these structural clues with your general software engineering knowledge to explain the architecture, concepts, or design intent. "
        "Clearly indicate what is derived directly from the code context versus what is inferred based on general programming practices.\n\n"
        "CRITICAL: You MUST write your detailed, step-by-step thinking process and code analysis inside <think> and </think> tags FIRST, "
        "and then write your final answer outside the tags. You must strictly follow this format:\n"
        "<think>\n"
        "[Your step-by-step reasoning, context analysis, and synthesis with general knowledge]\n"
        "</think>\n\n"
        "[Your final detailed answer]\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def _llm_generate(prompt: str, model: str) -> str:
    """Unified LLM generate helper supporting Ollama and Gemini with full tracing."""
    is_gemini = _is_gemini_provider(model, "LLM_PROVIDER")

    with start_span(
        "llm_generate", {"model": model, "provider": "gemini" if is_gemini else "ollama"}
    ):
        if is_gemini:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is not set. Please set GEMINI_API_KEY to use Gemini models, or switch to an Ollama model."
                )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0},
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                res_json = resp.json()

                usage = res_json.get("usageMetadata") or res_json.get("usage_metadata") or {}
                if usage:
                    prompt_tokens = (
                        usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0
                    )
                    candidates_tokens = (
                        usage.get("candidatesTokenCount")
                        or usage.get("candidates_token_count")
                        or 0
                    )
                    total_tokens = (
                        usage.get("totalTokenCount") or usage.get("total_token_count") or 0
                    )
                    logger.info(
                        "Gemini generation tokens: prompt=%d, completion=%d, total=%d",
                        prompt_tokens,
                        candidates_tokens,
                        total_tokens,
                    )

                candidates = res_json.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "5m")
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{_OLLAMA_URL}/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")


async def _llm_stream(prompt: str, model: str) -> AsyncIterator[str]:
    """Unified LLM streaming helper supporting Ollama and Gemini (yielding event-stream format)."""
    is_gemini = _is_gemini_provider(model, "LLM_PROVIDER")

    if is_gemini:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. Please set GEMINI_API_KEY to use Gemini models, or switch to an Ollama model."
            )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            token = parts[0].get("text", "")
                            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "5m")
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
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

    # 1. Check Redis Cache
    cache_key = RedisQueryCache.generate_key(
        question=req.question,
        codebase_query=req.codebase_query,
        repository=req.repository,
        llm_model=req.llm_model,
        top_k=req.top_k,
        context_n=req.context_n,
    )
    
    cached_resp = await app.state.query_cache.get(cache_key)
    if cached_resp is not None:
        logger.info("Serving query response from Redis cache: '%s'", req.question)
        cached_resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return QueryResponse(**cached_resp)

    analysis = analyze(req.question)

    if req.codebase_query:
        retriever: HybridRetriever = app.state.retriever
        with start_span(
            "api_query_endpoint", {"question": req.question, "repository": req.repository or "all"}
        ):
            try:
                # Route budget parameters
                with start_span(
                    "api_context_retrieval", {"top_k": req.top_k, "context_n": req.context_n}
                ):
                    if req.max_tokens is None and req.max_chars is None:
                        ctx = retriever.retrieve_with_context(
                            req.question,
                            top_k=req.top_k,
                            max_tokens=None,
                            max_chars=None,
                            context_n=req.context_n,
                            repository=req.repository,
                        )
                    else:
                        ctx = retriever.retrieve_with_context(
                            req.question,
                            top_k=req.top_k,
                            max_tokens=req.max_tokens,
                            max_chars=req.max_chars,
                            repository=req.repository,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Retrieval failed")
                raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc

            # Apply Dynamic Context Safety Cap to prevent VRAM overflow
            context_text = ctx.text or ""
            max_safe_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", 12000))
            if len(context_text) > max_safe_chars:
                logger.warning(
                    "Context size (%d chars) exceeds safety cap (%d chars). Truncating context.",
                    len(context_text),
                    max_safe_chars
                )
                context_text = (
                    context_text[:max_safe_chars] + 
                    "\n\n... [Context truncated to prevent VRAM overflow] ..."
                )

            prompt = _build_prompt(
                req.question,
                context_text,
                is_global=(analysis.query_type == "global"),
            )
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
            q_type = analysis.query_type
    else:
        prompt = (
            "You are an expert AI software developer and codebase assistant. "
            "CRITICAL: You MUST write your step-by-step thinking process and reasoning inside <think> and </think> tags FIRST, "
            "and then write your final answer outside the tags. You must strictly follow this format:\n"
            "<think>\n"
            "[Your thinking process]\n"
            "</think>\n\n"
            "[Your final answer]\n\n"
            f"Question: {req.question}\n\n"
            "Answer:"
        )
        sources = []
        q_type = "general"

    try:
        # Concurrency Guard
        async with app.state.llm_semaphore:
            answer = await _llm_generate(prompt, req.llm_model)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc
    finally:
        gc.collect()

    latency_ms = (time.perf_counter() - t0) * 1000

    response = QueryResponse(
        question=req.question,
        answer=answer,
        query_type=q_type,
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )
    
    try:
        await app.state.query_cache.set(cache_key, response.model_dump())
    except Exception as exc:
        logger.warning("Failed to cache response: %s", exc)

    return response


# ── POST /query/stream ─────────────────────────────────────────────────────────


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    Hybrid retrieval + LLM answer — Server-Sent Events stream.

    First event: {"token": "", "done": false, "sources": [...], "query_type": "..."}
    Subsequent: {"token": "<word>", "done": false}
    Final:       {"token": "", "done": true}
    """
    # 1. Check Redis Cache
    cache_key = RedisQueryCache.generate_key(
        question=req.question,
        codebase_query=req.codebase_query,
        repository=req.repository,
        llm_model=req.llm_model,
        top_k=req.top_k,
        context_n=req.context_n,
    )
    
    cached_events = await app.state.query_cache.get(cache_key)
    if cached_events is not None:
        logger.info("Serving query streaming response from Redis cache: '%s'", req.question)
        async def _cached_stream() -> AsyncIterator[str]:
            for event in cached_events:
                yield event
            gc.collect()
        return StreamingResponse(
            _cached_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    analysis = analyze(req.question)

    if req.codebase_query:
        retriever: HybridRetriever = app.state.retriever
        with start_span(
            "api_query_stream", {"question": req.question, "repository": req.repository or "all"}
        ):
            try:
                # Route budget parameters
                with start_span(
                    "api_context_retrieval_stream", {"top_k": req.top_k, "context_n": req.context_n}
                ):
                    if req.max_tokens is None and req.max_chars is None:
                        ctx = retriever.retrieve_with_context(
                            req.question,
                            top_k=req.top_k,
                            max_tokens=None,
                            max_chars=None,
                            context_n=req.context_n,
                            repository=req.repository,
                        )
                    else:
                        ctx = retriever.retrieve_with_context(
                            req.question,
                            top_k=req.top_k,
                            max_tokens=req.max_tokens,
                            max_chars=req.max_chars,
                            repository=req.repository,
                        )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc

            # Apply Dynamic Context Safety Cap to prevent VRAM overflow
            context_text = ctx.text or ""
            max_safe_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", 12000))
            if len(context_text) > max_safe_chars:
                logger.warning(
                    "Context size (%d chars) exceeds safety cap (%d chars). Truncating context.",
                    len(context_text),
                    max_safe_chars
                )
                context_text = (
                    context_text[:max_safe_chars] + 
                    "\n\n... [Context truncated to prevent VRAM overflow] ..."
                )

            prompt = _build_prompt(
                req.question,
                context_text,
                is_global=(analysis.query_type == "global"),
            )

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
            q_type = analysis.query_type
    else:
        prompt = (
            "You are an expert AI software developer and codebase assistant. "
            "CRITICAL: You MUST write your step-by-step thinking process and reasoning inside <think> and </think> tags FIRST, "
            "and then write your final answer outside the tags. You must strictly follow this format:\n"
            "<think>\n"
            "[Your thinking process]\n"
            "</think>\n\n"
            "[Your final answer]\n\n"
            f"Question: {req.question}\n\n"
            "Answer:"
        )
        sources_payload = []
        q_type = "general"

    async def _event_stream() -> AsyncIterator[str]:
        # First event: metadata
        meta = json.dumps(
            {
                "token": "",
                "done": False,
                "sources": sources_payload,
                "query_type": q_type,
            }
        )
        meta_event = f"data: {meta}\n\n"
        yield meta_event
        
        events_accumulated = [meta_event]

        try:
            # Concurrency Guard
            async with app.state.llm_semaphore:
                async for chunk in _llm_stream(prompt, req.llm_model):
                    yield chunk
                    events_accumulated.append(chunk)
            
            # Cache the successful stream
            try:
                await app.state.query_cache.set(cache_key, events_accumulated)
            except Exception as exc:
                logger.warning("Failed to cache stream response: %s", exc)
        except Exception as exc:
            logger.exception("Streaming LLM generation failed")
            error_data = json.dumps({"token": f"\n\n[Error: {exc}]", "done": True})
            yield f"data: {error_data}\n\n"
        finally:
            gc.collect()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /graph/repositories ───────────────────────────────────────────────────


@app.get("/graph/repositories")
async def list_repositories() -> list[str]:
    """Return all unique repository namespaces present in the graph database."""
    store: FalkorDBStore = app.state.graph_store
    try:
        return store.list_repositories()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list repositories: %s", exc)
        return []


@app.get("/graph/repositories/{repo_name}/status")
async def get_repository_status(repo_name: str) -> dict[str, Any]:
    """Retrieve indexing sync status and active background tasks for a repository."""
    store: FalkorDBStore = app.state.graph_store
    
    metadata = None
    try:
        metadata = store.get_repository_metadata(repo_name)
    except Exception as exc:
        logger.warning("Failed to get repository metadata: %s", exc)

    last_commit = metadata.get("last_indexed_commit") if metadata else None
    repo_path = metadata.get("repo_path") if metadata else None
    
    # Path fallback resolution inside Docker container
    effective_path = repo_path
    if repo_path and not os.path.isdir(repo_path):
        fallback_1 = os.path.join("/codebases", repo_name)
        if os.path.isdir(fallback_1):
            effective_path = fallback_1
        else:
            dir_name = os.path.basename(repo_path)
            fallback_2 = os.path.join("/codebases", dir_name)
            if os.path.isdir(fallback_2):
                effective_path = fallback_2

    # Check path accessibility status
    path_status = "valid"
    if not repo_path:
        path_status = "not_provided"
    elif not os.path.isdir(effective_path):
        path_status = "not_found"
    else:
        try:
            files = os.listdir(effective_path)
            if not files:
                path_status = "empty"
            elif not os.path.isdir(os.path.join(effective_path, ".git")):
                path_status = "not_a_git_repo"
        except Exception:
            path_status = "inaccessible"

    head_commit = None
    if path_status == "valid":
        try:
            import subprocess
            res_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=effective_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            head_commit = res_head.stdout.strip()
        except Exception:
            path_status = "inaccessible"

    is_sync = (last_commit is not None) and (head_commit is not None) and (last_commit == head_commit)
    
    active_task = None
    for task in app.state.indexing_tasks.values():
        if task["repository"] == repo_name and task["status"] in ("pending", "running"):
            active_task = {
                "task_id": task["task_id"],
                "status": task["status"]
            }
            break

    return {
        "repository": repo_name,
        "last_indexed_commit": last_commit,
        "repo_path": repo_path,
        "effective_path": effective_path,
        "head_commit": head_commit,
        "is_sync": is_sync,
        "path_status": path_status,
        "active_task": active_task,
    }


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

    return GraphNeighborsResponse(node_id=node_id, label=label, name=name, neighbors=edges)


# ── GET /graph/search ──────────────────────────────────────────────────────────


@app.get("/graph/search", response_model=GraphSearchResponse)
async def graph_search(
    q: str = Query(..., min_length=1, max_length=200),
    repository: str | None = Query(None, description="Scope query by repository name."),
    limit: int = Query(20, ge=1, le=50),
) -> GraphSearchResponse:
    """Find KG nodes whose name contains *q*."""
    store: FalkorDBStore = app.state.graph_store
    try:
        raw = store.find_nodes(q, repository=repository, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    nodes = [
        GraphNode(
            id=n.get("node_id", ""),
            name=n.get("name", ""),
            label=n.get("label", ""),
            file_path=n.get("file_path", ""),
            repository=n.get("repository", ""),
        )
        for n in raw
    ]
    return GraphSearchResponse(query=q, nodes=nodes)


# ── Background Indexing Worker & Endpoints ────────────────────────────────────


async def process_indexing_task(
    task_id: str,
    req: IndexRequest,
    app_state: Any,
) -> None:
    task = app_state.indexing_tasks[task_id]

    def add_log(msg: str) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task["logs"].append(f"[{timestamp}] {msg}")
        logger.info(f"Task {task_id}: {msg}")

    add_log("Waiting to acquire indexing lock...")

    async with app_state.indexing_lock:
        task["status"] = "running"
        add_log("Lock acquired. Starting indexing pipeline...")

        from hybrid_rag.ingestion.pipeline import IndexingListener, run_indexing_pipeline

        class ApiIndexingListener(IndexingListener):
            def on_step(self, step_name: str, message: str, progress: float | None = None) -> None:
                add_log(f"[{step_name}] {message}")

        try:
            # run_indexing_pipeline is synchronous; run in a thread pool to avoid blocking the main event loop
            result = await asyncio.to_thread(
                run_indexing_pipeline,
                repo_path=Path(req.repo_path),
                languages=req.languages,
                repo_name=task["repository"],
                graph_store=app_state.graph_store,
                vector_store=app_state.vector_store,
                ollama_url=_OLLAMA_URL,
                embed_model=_EMBED_MODEL,
                llm_model=os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL,
                llm_extract=req.llm_extract,
                max_tokens=req.max_tokens,
                listener=ApiIndexingListener(),
                incremental=req.incremental,
                rebuild=req.rebuild,
            )

            task["status"] = "completed"
            task["completed_at"] = datetime.datetime.now().isoformat()
            add_log(
                f"Indexing completed successfully. Elapsed: {result['elapsed_seconds']:.2f}s. "
                f"Nodes: {result['nodes_upserted']}, Edges: {result['edges_upserted']}, "
                f"Vectors: {result['vectors_upserted']}."
            )

        except Exception as exc:
            task["status"] = "failed"
            task["error"] = str(exc)
            task["completed_at"] = datetime.datetime.now().isoformat()
            add_log(f"Indexing failed with error: {exc}")
            logger.exception(f"Indexing task {task_id} failed")


@app.post("/graph/index", response_model=IndexTaskResponse)
async def trigger_index(
    req: IndexRequest,
    background_tasks: BackgroundTasks,
) -> IndexTaskResponse:
    """Queue a repository to be indexed in the background (serialized FIFO)."""
    # Verify path
    path = Path(req.repo_path)
    if not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Provided repo_path does not exist or is not a directory: {req.repo_path}",
        )

    task_id = str(uuid.uuid4())
    repo_name = req.repo_name or path.name

    task = {
        "task_id": task_id,
        "repository": repo_name,
        "status": "pending",
        "created_at": datetime.datetime.now().isoformat(),
        "completed_at": None,
        "logs": [
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task initialized and queued."
        ],
        "error": None,
    }
    app.state.indexing_tasks[task_id] = task

    # Queue background task
    background_tasks.add_task(process_indexing_task, task_id, req, app.state)

    return IndexTaskResponse(
        task_id=task_id,
        status="pending",
        repository=repo_name,
    )


@app.get("/graph/index/tasks", response_model=list[IndexTaskDetailResponse])
async def list_indexing_tasks() -> list[IndexTaskDetailResponse]:
    """List all indexing tasks queued or run in the background."""
    tasks = []
    for t in app.state.indexing_tasks.values():
        tasks.append(IndexTaskDetailResponse(**t))
    # Sort by created_at descending
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return tasks


@app.get("/graph/index/tasks/{task_id}", response_model=IndexTaskDetailResponse)
async def get_indexing_task(task_id: str) -> IndexTaskDetailResponse:
    """Retrieve detailed status and real-time logs for a specific indexing task."""
    if task_id not in app.state.indexing_tasks:
        raise HTTPException(status_code=404, detail=f"Indexing task not found: {task_id}")
    return IndexTaskDetailResponse(**app.state.indexing_tasks[task_id])
