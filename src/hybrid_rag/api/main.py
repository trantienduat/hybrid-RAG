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
_EMBED_MODEL = os.environ.get("EMBED_MODEL") or "nomic-embed-text"
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
            async with httpx.AsyncClient(timeout=120.0) as client:
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
            payload = {"model": model, "prompt": prompt, "stream": False}
            async with httpx.AsyncClient(timeout=120.0) as client:
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
        async with httpx.AsyncClient(timeout=120.0) as client:
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

        prompt = _build_prompt(
            req.question,
            ctx.text or "(no code context retrieved)",
            is_global=(analysis.query_type == "global"),
        )

        try:
            answer = await _llm_generate(prompt, req.llm_model)
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

        prompt = _build_prompt(
            req.question,
            ctx.text or "(no code context retrieved)",
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

        async def _event_stream() -> AsyncIterator[str]:
            # First event: metadata
            meta = json.dumps(
                {
                    "token": "",
                    "done": False,
                    "sources": sources_payload,
                    "query_type": analysis.query_type,
                }
            )
            yield f"data: {meta}\n\n"

            # Stream LLM tokens
            async for chunk in _llm_stream(prompt, req.llm_model):
                yield chunk

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
                llm_model=os.environ.get("LLM_MODEL") or "gemma4:12b",
                llm_extract=req.llm_extract,
                max_tokens=req.max_tokens,
                listener=ApiIndexingListener(),
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
