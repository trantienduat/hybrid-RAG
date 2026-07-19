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
import gc
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
from prometheus_client import Counter, Histogram, make_asgi_app

from hybrid_rag.api.schemas import (
    GraphNeighborsResponse,
    GraphNode,
    GraphSearchResponse,
    HealthResponse,
    IndexRequest,
    IndexTaskDetailResponse,
    IndexTaskResponse,
    LLMModelResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)


from hybrid_rag.config import translate_path_for_docker
from hybrid_rag.constants import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.retrieval.query_analyzer import analyze
from hybrid_rag.utils.cache import RedisQueryCache
from hybrid_rag.utils.tracing import initialize_tracing, start_span
from hybrid_rag.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Define Prometheus metrics
RAG_TOKENS_SAVED = Counter(
    "rag_tokens_saved_total",
    "Total input tokens saved by using RAG instead of full codebase context",
    ["model", "query_type"],
)
QUERY_CACHE_HITS = Counter(
    "query_cache_hits_total", "Total number of query hits resolved from Redis cache"
)
LLM_TOKENS_CONSUMED = Counter(
    "llm_tokens_consumed_total",
    "Total tokens consumed by the LLM backend",
    ["model", "token_type"],
)
QUERY_DURATION = Histogram(
    "query_duration_seconds",
    "Time taken to resolve the query",
    ["query_type", "cache_status"],
)

_STATIC_DIR = Path(__file__).parent / "static"


def _is_gemini_provider(model: str, provider_env_var: str | None = None) -> bool:
    """Check if the given model or environment override indicates Gemini provider."""
    if model.startswith("gemini") or model == "text-embedding-004":
        return True
    if provider_env_var and os.environ.get(provider_env_var) == "gemini":
        return True
    return False


# ── Configuration from environment/config file ──────────────────────────────────

from hybrid_rag.config import app_config

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



async def _run_periodic_sync(app_state: Any) -> None:
    """Periodic background worker that runs incremental sync for all repositories."""
    enabled = os.environ.get("CRON_SYNC_ENABLED", "false").lower() == "true"
    if not enabled:
        logger.info("Scheduled periodic sync is disabled.")
        return

    interval_min = int(os.environ.get("CRON_SYNC_INTERVAL_MINUTES", "60"))
    logger.info("Starting periodic sync loop: every %d minutes", interval_min)

    # Startup delay: wait 60s
    await asyncio.sleep(60)

    while True:
        try:
            logger.info("Running scheduled repository synchronization check...")
            db_repos = []
            try:
                db_repos = app_state.graph_store.list_repositories()
            except Exception:
                pass
            config_repos = [r["name"] for r in app_config.repositories]
            repos = sorted(list(set(db_repos + config_repos)))

            for repo_name in repos:
                metadata = None
                try:
                    metadata = app_state.graph_store.get_repository_metadata(repo_name)
                except Exception:
                    pass

                repo_path = app_config.get_repo_path(repo_name)
                if not repo_path:
                    continue

                last_commit = metadata.get("last_indexed_commit") if metadata else None

                # Resolve effective path in container
                effective_path = translate_path_for_docker(repo_path)

                # Verify directory exists and is a git repository
                if not os.path.isdir(effective_path):
                    continue
                curr = Path(effective_path)
                is_git = False
                while True:
                    if (curr / ".git").is_dir():
                        is_git = True
                        break
                    if curr == curr.parent:
                        break
                    curr = curr.parent
                if not is_git:
                    continue

                # Run fast Git check (rev-parse HEAD) to see if we actually need to sync
                head_commit = None
                try:
                    import subprocess

                    res_head = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=effective_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    head_commit = res_head.stdout.strip()
                except Exception:
                    continue

                # If commit hash is identical, skip creating a task! Save CPU/VRAM/Task history
                if last_commit == head_commit:
                    logger.debug(
                        "Scheduled sync: Repository %s is already up-to-date at %s, skipping",
                        repo_name,
                        head_commit,
                    )
                    continue

                # Avoid duplicate tasks if one is already running
                already_running = False
                for task in app_state.indexing_tasks.values():
                    if task["repository"] == repo_name and task["status"] in ("pending", "running"):
                        already_running = True
                        break

                if already_running:
                    continue

                # Trigger incremental index task under background worker
                task_id = str(uuid.uuid4())
                app_state.indexing_tasks[task_id] = {
                    "task_id": task_id,
                    "repository": repo_name,
                    "status": "pending",
                    "created_at": datetime.datetime.now().isoformat(),
                    "completed_at": None,
                    "logs": [
                        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scheduled periodic sync task initialized."
                    ],
                    "error": None,
                }

                # Import IndexRequest locally
                from hybrid_rag.api.schemas import IndexRequest

                req = IndexRequest(
                    repo_path=effective_path, repo_name=repo_name, incremental=True, rebuild=False
                )

                # process_indexing_task is async, we run it as an independent task
                asyncio.create_task(process_indexing_task(task_id, req, app_state))

        except Exception as exc:
            logger.exception("Scheduled sync loop encountered an error: %s", exc)

        await asyncio.sleep(interval_min * 60)


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

    # Initialize global HTTP client
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    # Initialize Concurrency Guard and Redis Query Cache
    app.state.llm_semaphore = asyncio.Semaphore(1)
    app.state.query_cache = RedisQueryCache()
    await app.state.query_cache.connect()

    # Start scheduled periodic sync background task
    asyncio.create_task(_run_periodic_sync(app.state))

    logger.info("hybrid-rag API ready")
    yield
    app.state.embedder.close()
    await app.state.query_cache.disconnect()
    await app.state.http_client.aclose()
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

app.mount("/metrics", make_asgi_app())

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


def _is_vietnamese(text: str) -> bool:
    import re

    # 1. Check diacritics
    vietnamese_diacritics = re.compile(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    )
    if vietnamese_diacritics.search(text):
        return True

    # 2. Check common non-diacritic Vietnamese words/stop words with low English overlap
    viet_words = {
        "luong",
        "cach",
        "tong",
        "hop",
        "giai",
        "thich",
        "huong",
        "dan",
        "phan",
        "tich",
        "viet",
        "khong",
        "cua",
        "cai",
        "nao",
        "sao",
        "lam",
        "chay",
        "chuc",
        "nang",
        "hoat",
        "dong",
        "thuc",
        "dung",
        "tieng",
        "xem",
    }
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if words & viet_words:
        return True

    return False


def _is_reasoning_model(model: str | None) -> bool:
    """Detect if the model is a reasoning/thinking model (like deepseek-r1)."""
    if not model:
        return False
    model_lower = model.lower()
    return "r1" in model_lower or "reasoner" in model_lower or "thinking" in model_lower


def _estimate_num_ctx(prompt: str) -> int:
    """Estimate dynamic context window size (num_ctx) for Ollama based on prompt length."""
    env_ctx = os.environ.get("OLLAMA_NUM_CTX")
    if env_ctx:
        try:
            return int(env_ctx)
        except ValueError:
            pass

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoding.encode(prompt))
    except Exception:
        # Fallback approximation: 1 token ~= 4 characters
        prompt_tokens = len(prompt) // 4

    # We want at least 2048 tokens of response buffer
    target_ctx = prompt_tokens + 2048

    # Clamp between a safe minimum (8192) and a safe maximum (32768)
    min_ctx = 8192
    max_ctx = int(os.environ.get("OLLAMA_MAX_NUM_CTX", "32768"))

    # Round up to nearest 1024
    target_ctx = ((target_ctx + 1023) // 1024) * 1024
    return max(min_ctx, min(target_ctx, max_ctx))


def _build_prompt(
    question: str, context: str, model: str | None = None, is_global: bool = False
) -> str:
    is_viet = _is_vietnamese(question)
    use_thinking = _is_reasoning_model(model)

    if is_viet:
        if use_thinking:
            lang_instruction = (
                "IMPORTANT: The user's question is in Vietnamese. You MUST think and answer in Vietnamese. "
                "Both the content inside <think>...</think> and the final answer MUST be written entirely in Vietnamese.\n"
                "LƯU Ý QUAN TRỌNG: Câu hỏi của người dùng bằng Tiếng Việt. Bạn PHẢI suy nghĩ trong <think> và trả lời bằng Tiếng Việt. "
                "Tất cả nội dung suy nghĩ và câu trả lời cuối cùng đều phải viết bằng Tiếng Việt."
            )
            if is_global:
                return (
                    "You are an expert principal software architect. Below is a set of hierarchical community summaries "
                    "describing the structural design, modules, and dependencies of the codebase.\n"
                    "Analyze these summaries and provide a comprehensive, highly-structured architectural report. "
                    "Highlight key components, database models, core flows, and cross-module relationships.\n"
                    "If the summaries are sparse, combine these structural clues with your general software architecture knowledge "
                    "to infer design patterns, architectures, and intent.\n"
                    f"{lang_instruction}\n\n"
                    "CRITICAL: You MUST write your detailed, step-by-step reasoning process inside <think> and </think> tags FIRST, "
                    "and then write your final report outside the tags. You must strictly follow this format:\n"
                    "<think>\n"
                    "[Viết quá trình suy nghĩ và phân tích chi tiết của bạn tại đây bằng Tiếng Việt]\n"
                    "</think>\n\n"
                    "[Viết báo cáo kiến trúc cuối cùng của bạn tại đây bằng Tiếng Việt]\n\n"
                    f"Community Summaries Context:\n{context}\n\n"
                    f"User Request: {question}\n\n"
                    f"{lang_instruction}\n"
                    "Architectural Report (in Vietnamese):"
                )
            return (
                "You are an expert code assistant. Use the provided context below as the primary source of truth to answer the question.\n"
                "If the context is sparse (e.g. only contains a list of directories, modules, or file definitions) but lacks conceptual detail, "
                "you should synthesize these structural clues with your general software engineering knowledge to explain the architecture, concepts, or design intent. "
                "Clearly indicate what is derived directly from the code context versus what is inferred based on general programming practices.\n"
                f"{lang_instruction}\n\n"
                "CRITICAL: You MUST write your detailed, step-by-step thinking process and code analysis inside <think> and </think> tags FIRST, "
                "and then write your final answer outside the tags. You must strictly follow this format:\n"
                "<think>\n"
                "[Viết quá trình suy nghĩ và phân tích của bạn tại đây bằng Tiếng Việt]\n"
                "</think>\n\n"
                "[Viết câu trả lời chi tiết cuối cùng của bạn tại đây bằng Tiếng Việt]\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"{lang_instruction}\n"
                "Answer (in Vietnamese):"
            )
        else:
            lang_instruction = (
                "IMPORTANT: The user's question is in Vietnamese. You MUST answer in Vietnamese.\n"
                "LƯU Ý QUAN TRỌNG: Câu hỏi của người dùng bằng Tiếng Việt. Bạn PHẢI trả lời bằng Tiếng Việt."
            )
            if is_global:
                return (
                    "You are an expert principal software architect. Below is a set of hierarchical community summaries "
                    "describing the structural design, modules, and dependencies of the codebase.\n"
                    "Analyze these summaries and provide a comprehensive, highly-structured architectural report. "
                    "Highlight key components, database models, core flows, and cross-module relationships.\n"
                    "If the summaries are sparse, combine these structural clues with your general software architecture knowledge "
                    "to infer design patterns, architectures, and intent.\n"
                    f"{lang_instruction}\n\n"
                    f"Community Summaries Context:\n{context}\n\n"
                    f"User Request: {question}\n\n"
                    f"{lang_instruction}\n"
                    "Architectural Report (in Vietnamese):"
                )
            return (
                "You are an expert code assistant. Use the provided context below as the primary source of truth to answer the question.\n"
                "If the context is sparse (e.g. only contains a list of directories, modules, or file definitions) but lacks conceptual detail, "
                "you should synthesize these structural clues with your general software engineering knowledge to explain the architecture, concepts, or design intent. "
                "Clearly indicate what is derived directly from the code context versus what is inferred based on general programming practices.\n"
                f"{lang_instruction}\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"{lang_instruction}\n"
                "Answer (in Vietnamese):"
            )

    if use_thinking:
        if is_global:
            return (
                "You are an expert principal software architect. Below is a set of hierarchical community summaries "
                "describing the structural design, modules, and dependencies of the codebase.\n"
                "Analyze these summaries and provide a comprehensive, highly-structured architectural report. "
                "Highlight key components, database models, core flows, and cross-module relationships.\n"
                "If the summaries are sparse, combine these structural clues with your general software architecture knowledge "
                "to infer design patterns, architectures, and intent.\n"
                "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
                "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n\n"
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
            "Clearly indicate what is derived directly from the code context versus what is inferred based on general programming practices.\n"
            "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
            "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n\n"
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
    else:
        if is_global:
            return (
                "You are an expert principal software architect. Below is a set of hierarchical community summaries "
                "describing the structural design, modules, and dependencies of the codebase.\n"
                "Analyze these summaries and provide a comprehensive, highly-structured architectural report. "
                "Highlight key components, database models, core flows, and cross-module relationships.\n"
                "If the summaries are sparse, combine these structural clues with your general software architecture knowledge "
                "to infer design patterns, architectures, and intent.\n"
                "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
                "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n\n"
                f"Community Summaries Context:\n{context}\n\n"
                f"User Request: {question}\n\n"
                "Architectural Report:"
            )
        return (
            "You are an expert code assistant. Use the provided context below as the primary source of truth to answer the question.\n"
            "If the context is sparse (e.g. only contains a list of directories, modules, or file definitions) but lacks conceptual detail, "
            "you should synthesize these structural clues with your general software engineering knowledge to explain the architecture, concepts, or design intent. "
            "Clearly indicate what is derived directly from the code context versus what is inferred based on general programming practices.\n"
            "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
            "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n\n"
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
        client = app.state.http_client
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
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            res_json = resp.json()

            usage = res_json.get("usageMetadata") or res_json.get("usage_metadata") or {}
            if usage:
                prompt_tokens = (
                    usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0
                )
                candidates_tokens = (
                    usage.get("candidatesTokenCount") or usage.get("candidates_token_count") or 0
                )
                total_tokens = usage.get("totalTokenCount") or usage.get("total_token_count") or 0
                logger.info(
                    "Gemini generation tokens: prompt=%d, completion=%d, total=%d",
                    prompt_tokens,
                    candidates_tokens,
                    total_tokens,
                )
                if prompt_tokens > 0:
                    LLM_TOKENS_CONSUMED.labels(model=model, token_type="prompt").inc(prompt_tokens)
                if candidates_tokens > 0:
                    LLM_TOKENS_CONSUMED.labels(model=model, token_type="completion").inc(
                        candidates_tokens
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
                "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "5m"),
                "options": {"num_ctx": _estimate_num_ctx(prompt)},
            }
            resp = await client.post(f"{_OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            res_json = resp.json()
            prompt_tokens = res_json.get("prompt_eval_count", 0)
            candidates_tokens = res_json.get("eval_count", 0)
            if prompt_tokens > 0:
                LLM_TOKENS_CONSUMED.labels(model=model, token_type="prompt").inc(prompt_tokens)
            if candidates_tokens > 0:
                LLM_TOKENS_CONSUMED.labels(model=model, token_type="completion").inc(
                    candidates_tokens
                )
            return res_json.get("response", "")


async def _llm_stream(prompt: str, model: str) -> AsyncIterator[str]:
    """Unified LLM streaming helper supporting Ollama and Gemini (yielding event-stream format)."""
    is_gemini = _is_gemini_provider(model, "LLM_PROVIDER")
    client = app.state.http_client

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
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                usage = data.get("usageMetadata") or data.get("usage_metadata")
                if usage:
                    prompt_tokens = (
                        usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0
                    )
                    candidates_tokens = (
                        usage.get("candidatesTokenCount")
                        or usage.get("candidates_token_count")
                        or 0
                    )
                    if prompt_tokens > 0:
                        LLM_TOKENS_CONSUMED.labels(model=model, token_type="prompt").inc(
                            prompt_tokens
                        )
                    if candidates_tokens > 0:
                        LLM_TOKENS_CONSUMED.labels(model=model, token_type="completion").inc(
                            candidates_tokens
                        )

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
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "5m"),
            "options": {"num_ctx": _estimate_num_ctx(prompt)},
        }
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
                if data.get("done", False):
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    candidates_tokens = data.get("eval_count", 0)
                    if prompt_tokens > 0:
                        LLM_TOKENS_CONSUMED.labels(model=model, token_type="prompt").inc(
                            prompt_tokens
                        )
                    if candidates_tokens > 0:
                        LLM_TOKENS_CONSUMED.labels(model=model, token_type="completion").inc(
                            candidates_tokens
                        )
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


@app.get("/llm/models", response_model=list[LLMModelResponse])
async def list_models() -> list[LLMModelResponse]:
    """Retrieve the list of available local Ollama models, filtering out embeddings."""
    try:
        from hybrid_rag.constants import DEFAULT_LLM_MODEL

        client = app.state.http_client
        resp = await client.get(f"{_OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        models = resp.json().get("models", [])

        result = []
        for m in models:
            name = m.get("name", "")
            # Filter out embedding models
            if "embed" in name.lower():
                continue
            details = m.get("details", {})

            # Match either exact name or base name (e.g. qwen2.5-coder:7b vs qwen2.5-coder:latest)
            is_default = (
                name == DEFAULT_LLM_MODEL or name.split(":")[0] == DEFAULT_LLM_MODEL.split(":")[0]
            )

            result.append(
                LLMModelResponse(
                    name=name,
                    parameter_size=details.get("parameter_size"),
                    size_bytes=m.get("size"),
                    is_default=is_default,
                )
            )
        return result
    except Exception:
        logger.exception("Failed to fetch Ollama models")
        from hybrid_rag.constants import DEFAULT_LLM_MODEL

        return [
            LLMModelResponse(name=DEFAULT_LLM_MODEL, is_default=True),
        ]


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
        QUERY_CACHE_HITS.inc()
        QUERY_DURATION.labels(
            query_type=cached_resp.get("query_type", "general"), cache_status="hit"
        ).observe(time.perf_counter() - t0)
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
                    max_safe_chars,
                )
                context_text = (
                    context_text[:max_safe_chars]
                    + "\n\n... [Context truncated to prevent VRAM overflow] ..."
                )

            prompt = _build_prompt(
                req.question,
                context_text,
                model=req.llm_model,
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
        is_viet = _is_vietnamese(req.question)
        use_thinking = _is_reasoning_model(req.llm_model)
        if is_viet:
            if use_thinking:
                lang_instruction = (
                    "IMPORTANT: The user's question is in Vietnamese. You MUST think and answer in Vietnamese. "
                    "Both the content inside <think>...</think> and the final answer MUST be written entirely in Vietnamese.\n"
                    "LƯU Ý QUAN TRỌNG: Câu hỏi của người dùng bằng Tiếng Việt. Bạn PHẢI suy nghĩ trong <think> và trả lời bằng Tiếng Việt. "
                    "Tất cả nội dung suy nghĩ và câu trả lời cuối cùng đều phải viết bằng Tiếng Việt."
                )
                prompt = (
                    "You are an expert AI software developer and codebase assistant.\n"
                    f"{lang_instruction}\n\n"
                    "CRITICAL: You MUST write your step-by-step thinking process and reasoning inside <think> and </think> tags FIRST, "
                    "and then write your final answer outside the tags. You must strictly follow this format:\n"
                    "<think>\n"
                    "[Viết quá trình suy nghĩ và phân tích của bạn tại đây bằng Tiếng Việt]\n"
                    "</think>\n\n"
                    "[Viết câu trả lời cuối cùng của bạn tại đây bằng Tiếng Việt]\n\n"
                    f"Question: {req.question}\n\n"
                    f"{lang_instruction}\n"
                    "Answer (in Vietnamese):"
                )
            else:
                lang_instruction = (
                    "IMPORTANT: The user's question is in Vietnamese. You MUST answer in Vietnamese.\n"
                    "LƯU Ý QUAN TRỌNG: Câu hỏi của người dùng bằng Tiếng Việt. Bạn PHẢI trả lời bằng Tiếng Việt."
                )
                prompt = (
                    "You are an expert AI software developer and codebase assistant.\n"
                    f"{lang_instruction}\n\n"
                    f"Question: {req.question}\n\n"
                    f"{lang_instruction}\n"
                    "Answer (in Vietnamese):"
                )
        else:
            if use_thinking:
                prompt = (
                    "You are an expert AI software developer and codebase assistant. "
                    "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
                    "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n"
                    "CRITICAL: You MUST write your step-by-step thinking process and reasoning inside <think> and </think> tags FIRST, "
                    "and then write your final answer outside the tags. You must strictly follow this format:\n"
                    "<think>\n"
                    "[Your thinking process]\n"
                    "</think>\n\n"
                    "[Your final answer]\n\n"
                    f"Question: {req.question}\n\n"
                    "Answer:"
                )
            else:
                prompt = (
                    "You are an expert AI software developer and codebase assistant. "
                    "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
                    "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n\n"
                    f"Question: {req.question}\n\n"
                    "Answer:"
                )
        sources = []
        q_type = "general"

    t_llm = time.perf_counter()
    try:
        # Concurrency Guard
        async with app.state.llm_semaphore:
            answer = await _llm_generate(prompt, req.llm_model)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc
    finally:
        gc.collect()

    llm_ms = (time.perf_counter() - t_llm) * 1000
    latency_ms = (time.perf_counter() - t0) * 1000

    timings = {}
    if req.codebase_query:
        timings = ctx.timings.copy()
        timings["llm_ms"] = round(llm_ms, 2)
        timings["total_ms"] = round(latency_ms, 2)
    else:
        timings = {
            "parse_query_ms": 0.0,
            "graph_search_ms": 0.0,
            "vector_search_ms": 0.0,
            "rrf_ms": 0.0,
            "llm_ms": round(llm_ms, 2),
            "total_ms": round(latency_ms, 2),
        }

    response = QueryResponse(
        question=req.question,
        answer=answer,
        query_type=q_type,
        sources=sources,
        latency_ms=round(latency_ms, 2),
        timings=timings,
    )

    QUERY_DURATION.labels(query_type=q_type, cache_status="miss").observe(time.perf_counter() - t0)

    if req.codebase_query:
        try:
            import tiktoken

            try:
                count_res = app.state.vector_store._client.count(
                    collection_name=app.state.vector_store._collection, exact=True
                )
                codebase_tokens = count_res.count * 300
            except Exception:
                codebase_tokens = 160000

            prompt_tokens = len(tiktoken.get_encoding("cl100k_base").encode(prompt))
            saved_tokens = max(0, codebase_tokens - prompt_tokens)
            RAG_TOKENS_SAVED.labels(model=req.llm_model, query_type=q_type).inc(saved_tokens)
        except Exception as e:
            logger.warning("Failed to count RAG tokens saved: %s", e)

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

    cached_events = await app.state.query_cache.get(cache_key)
    if cached_events is not None:
        logger.info("Serving query streaming response from Redis cache: '%s'", req.question)
        QUERY_CACHE_HITS.inc()
        try:
            first_event = json.loads(cached_events[0].replace("data: ", "").strip())
            q_type = first_event.get("query_type", "general")
        except Exception:
            q_type = "general"
        QUERY_DURATION.labels(query_type=q_type, cache_status="hit").observe(
            time.perf_counter() - t0
        )

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
                    max_safe_chars,
                )
                context_text = (
                    context_text[:max_safe_chars]
                    + "\n\n... [Context truncated to prevent VRAM overflow] ..."
                )

            prompt = _build_prompt(
                req.question,
                context_text,
                model=req.llm_model,
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
        is_viet = _is_vietnamese(req.question)
        use_thinking = _is_reasoning_model(req.llm_model)
        if is_viet:
            if use_thinking:
                lang_instruction = (
                    "IMPORTANT: The user's question is in Vietnamese. You MUST think and answer in Vietnamese. "
                    "Both the content inside <think>...</think> and the final answer MUST be written entirely in Vietnamese.\n"
                    "LƯU Ý QUAN TRỌNG: Câu hỏi của người dùng bằng Tiếng Việt. Bạn PHẢI suy nghĩ trong <think> và trả lời bằng Tiếng Việt. "
                    "Tất cả nội dung suy nghĩ và câu trả lời cuối cùng đều phải viết bằng Tiếng Việt."
                )
                prompt = (
                    "You are an expert AI software developer and codebase assistant.\n"
                    f"{lang_instruction}\n\n"
                    "CRITICAL: You MUST write your step-by-step thinking process and reasoning inside <think> and </think> tags FIRST, "
                    "and then write your final answer outside the tags. You must strictly follow this format:\n"
                    "<think>\n"
                    "[Viết quá trình suy nghĩ và phân tích của bạn tại đây bằng Tiếng Việt]\n"
                    "</think>\n\n"
                    "[Viết câu trả lời cuối cùng của bạn tại đây bằng Tiếng Việt]\n\n"
                    f"Question: {req.question}\n\n"
                    f"{lang_instruction}\n"
                    "Answer (in Vietnamese):"
                )
            else:
                lang_instruction = (
                    "IMPORTANT: The user's question is in Vietnamese. You MUST answer in Vietnamese.\n"
                    "LƯU Ý QUAN TRỌNG: Câu hỏi của người dùng bằng Tiếng Việt. Bạn PHẢI trả lời bằng Tiếng Việt."
                )
                prompt = (
                    "You are an expert AI software developer and codebase assistant.\n"
                    f"{lang_instruction}\n\n"
                    f"Question: {req.question}\n\n"
                    f"{lang_instruction}\n"
                    "Answer (in Vietnamese):"
                )
        else:
            if use_thinking:
                prompt = (
                    "You are an expert AI software developer and codebase assistant. "
                    "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
                    "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n"
                    "CRITICAL: You MUST write your step-by-step thinking process and reasoning inside <think> and </think> tags FIRST, "
                    "and then write your final answer outside the tags. You must strictly follow this format:\n"
                    "<think>\n"
                    "[Your thinking process]\n"
                    "</think>\n\n"
                    "[Your final answer]\n\n"
                    f"Question: {req.question}\n\n"
                    "Answer:"
                )
            else:
                prompt = (
                    "You are an expert AI software developer and codebase assistant. "
                    "IMPORTANT: Always respond in the SAME LANGUAGE as the user's question. "
                    "If the user asks in Vietnamese, respond in Vietnamese. If in English, respond in English.\n\n"
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
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:].strip())
                            if data.get("done", False):
                                t_total = (time.perf_counter() - t0) * 1000
                                llm_ms = t_total - sum(ctx.timings.values()) if req.codebase_query else t_total
                                timings = {}
                                if req.codebase_query:
                                    timings = ctx.timings.copy()
                                    timings["llm_ms"] = round(max(0.0, llm_ms), 2)
                                    timings["total_ms"] = round(t_total, 2)
                                else:
                                    timings = {
                                        "parse_query_ms": 0.0,
                                        "graph_search_ms": 0.0,
                                        "vector_search_ms": 0.0,
                                        "rrf_ms": 0.0,
                                        "llm_ms": round(t_total, 2),
                                        "total_ms": round(t_total, 2),
                                    }
                                data["timings"] = timings
                                chunk = f"data: {json.dumps(data)}\n\n"
                        except Exception as e:
                            logger.warning("Failed to inject timings into SSE chunk: %s", e)
                    yield chunk
                    events_accumulated.append(chunk)

            QUERY_DURATION.labels(query_type=q_type, cache_status="miss").observe(
                time.perf_counter() - t0
            )

            if req.codebase_query:
                try:
                    import tiktoken

                    try:
                        count_res = app.state.vector_store._client.count(
                            collection_name=app.state.vector_store._collection, exact=True
                        )
                        codebase_tokens = count_res.count * 300
                    except Exception:
                        codebase_tokens = 160000

                    prompt_tokens = len(tiktoken.get_encoding("cl100k_base").encode(prompt))
                    saved_tokens = max(0, codebase_tokens - prompt_tokens)
                    RAG_TOKENS_SAVED.labels(model=req.llm_model, query_type=q_type).inc(
                        saved_tokens
                    )
                except Exception as e:
                    logger.warning("Failed to count RAG tokens saved: %s", e)

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


@app.get("/graph/master")
async def get_master_graph() -> dict[str, Any]:
    """Retrieve high-level RepositoryMetadata and Directory nodes and their relationships for global visualization."""
    store: FalkorDBStore = app.state.graph_store
    nodes = []
    links = []

    try:
        # 1. Fetch RepositoryMetadata nodes
        res_repos = store.query("MATCH (r:RepositoryMetadata) RETURN r")
        for row in res_repos.result_set or []:
            r_node = row[0]
            r_props = getattr(r_node, "properties", {})
            nodes.append({
                "id": r_props.get("id", ""),
                "label": "RepositoryMetadata",
                "name": r_props.get("id", ""),
                "repository": r_props.get("id", ""),
            })

        # 2. Fetch Directory nodes
        res_dirs = store.query("MATCH (d:Directory) RETURN d")
        for row in res_dirs.result_set or []:
            d_node = row[0]
            d_props = getattr(d_node, "properties", {})
            nodes.append({
                "id": d_props.get("id", ""),
                "label": "Directory",
                "name": d_props.get("name", ""),
                "repository": d_props.get("repository", ""),
                "path": d_props.get("path", ""),
            })

        # 3. Fetch WEAK_LINKs between Directory and RepositoryMetadata
        res_links_repo = store.query("MATCH (d:Directory)-[r:WEAK_LINK]->(m:RepositoryMetadata) RETURN d.id, m.id")
        for row in res_links_repo.result_set or []:
            links.append({
                "source": row[0],
                "target": row[1],
                "rel": "WEAK_LINK"
            })

        # 4. Fetch WEAK_LINKs between Directory and Directory
        res_links_dir = store.query("MATCH (d1:Directory)-[r:WEAK_LINK]->(d2:Directory) RETURN d1.id, d2.id")
        for row in res_links_dir.result_set or []:
            links.append({
                "source": row[0],
                "target": row[1],
                "rel": "WEAK_LINK"
            })
    except Exception as exc:
        logger.error("Failed to build master graph: %s", exc)

    return {"nodes": nodes, "links": links}


@app.get("/graph/repositories")
async def list_repositories() -> list[str]:
    """Return all unique repository namespaces present in the config or database."""
    from hybrid_rag.config import app_config
    configured = [r["name"] for r in app_config.repositories]

    store: FalkorDBStore = app.state.graph_store
    db_repos = []
    try:
        db_repos = store.list_repositories()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list repositories from database: %s", exc)

    return sorted(list(set(configured + db_repos)))


@app.get("/graph/repositories/{repo_name}/community")
async def get_repo_community_graph(repo_name: str) -> dict[str, Any]:
    """Retrieve community nodes and their dependencies for a specific repository."""
    store: FalkorDBStore = app.state.graph_store
    nodes = []
    links = []

    try:
        # Find all communities that have member nodes belonging to this repository
        cypher_nodes = (
            "MATCH (n)-[:IN_COMMUNITY]->(c:Community) "
            "WHERE n.repository = $repo_name "
            "RETURN DISTINCT c"
        )
        res_nodes = store.query(cypher_nodes, {"repo_name": repo_name})
        comm_ids = set()
        for row in res_nodes.result_set or []:
            c_node = row[0]
            c_props = getattr(c_node, "properties", {})
            c_id = c_props.get("id", "")
            if c_id and c_id not in comm_ids:
                comm_ids.add(c_id)
                nodes.append({
                    "id": c_id,
                    "label": "Community",
                    "name": c_props.get("name", f"Community {c_id}"),
                    "summary": c_props.get("summary", ""),
                    "repository": repo_name,
                })

        # Find relationships between these communities
        if comm_ids:
            cypher_links = (
                "MATCH (c1:Community)-[r:COMMUNITY_DEPENDS]->(c2:Community) "
                "RETURN c1.id, c2.id, r.weight"
            )
            res_links = store.query(cypher_links)
            for row in res_links.result_set or []:
                src_id, dst_id, weight = row[0], row[1], row[2]
                if src_id in comm_ids and dst_id in comm_ids:
                    links.append({
                        "source": src_id,
                        "target": dst_id,
                        "rel": f"COMMUNITY_DEPENDS (weight: {weight})",
                    })
    except Exception as exc:
        logger.error("Failed to build repository community graph: %s", exc)

    return {"nodes": nodes, "links": links}


@app.get("/graph/repositories/{repo_name}/status")
async def get_repository_status(repo_name: str) -> dict[str, Any]:
    """Retrieve indexing sync status and active background tasks for a repository."""
    from hybrid_rag.config import app_config
    store: FalkorDBStore = app.state.graph_store

    metadata = None
    try:
        metadata = store.get_repository_metadata(repo_name)
    except Exception as exc:
        logger.warning("Failed to get repository metadata: %s", exc)

    last_commit = metadata.get("last_indexed_commit") if metadata else None
    updated_at = metadata.get("updated_at") if metadata else None
    repo_path = app_config.get_repo_path(repo_name)

    last_synced = None
    if updated_at:
        try:
            last_synced = datetime.datetime.fromtimestamp(updated_at / 1000.0).isoformat()
        except Exception:
            pass

    effective_path = translate_path_for_docker(repo_path)

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
            else:
                curr = Path(effective_path)
                is_git = False
                while True:
                    if os.path.isdir(curr / ".git"):
                        is_git = True
                        break
                    if curr == curr.parent:
                        break
                    curr = curr.parent
                if not is_git:
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
                capture_output=True,
                text=True,
                check=True,
            )
            head_commit = res_head.stdout.strip()
        except Exception as exc:
            logger.warning("Failed to run git rev-parse HEAD in %s: %s", effective_path, exc, exc_info=True)
            path_status = "inaccessible"

    is_sync = (
        (last_commit is not None) and (head_commit is not None) and (last_commit == head_commit)
    )

    active_task = None
    for task in app.state.indexing_tasks.values():
        if task["repository"] == repo_name and task["status"] in ("pending", "running"):
            active_task = {
                "task_id": task["task_id"],
                "status": task["status"],
                "progress": task.get("progress", 0.0),
                "current_step": task.get("current_step", ""),
                "current_message": task.get("current_message", ""),
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
        "last_synced": last_synced,
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

    label = ""
    name = simple_name
    try:
        res = store.query(
            "MATCH (n) WHERE n.id = $id RETURN labels(n)[0] AS label, n.name AS name",
            {"id": node_id}
        )
        if res.result_set:
            row = res.result_set[0]
            label = row[0] or ""
            name = row[1] or simple_name
    except Exception as exc:
        logger.warning("Exact lookup failed for %s: %s", node_id, exc)

    if not label:
        try:
            nodes = store.find_nodes(simple_name, limit=5)
            anchor = next((n for n in nodes if n.get("node_id") == node_id), None)
            if anchor:
                label = anchor.get("label", "")
                name = anchor.get("name", simple_name)
        except Exception:
            pass

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

    if task.get("status") == "aborted":
        add_log("Task was aborted before it could run.")
        return

    async with app_state.indexing_lock:
        if task.get("status") == "aborted":
            add_log("Task was aborted before it could run.")
            return

        task["status"] = "running"
        add_log("Lock acquired. Starting indexing pipeline...")

        from hybrid_rag.ingestion.pipeline import IndexingListener, run_indexing_pipeline

        class ApiIndexingListener(IndexingListener):
            def on_step(self, step_name: str, message: str, progress: float | None = None) -> None:
                if app_state.indexing_tasks[task_id].get("status") == "aborted":
                    raise RuntimeError("Task aborted by user")

                add_log(f"[{step_name}] {message}")

                # Linear progress mapping
                p_val = progress if progress is not None else 0.0
                if step_name == "init":
                    overall = 0.05
                elif step_name == "git_diff":
                    overall = 0.05 + p_val * 0.05
                elif step_name == "parse":
                    overall = 0.10 + p_val * 0.30
                elif step_name == "llm_extract":
                    overall = 0.40 + p_val * 0.20
                elif step_name == "resolution":
                    overall = 0.60 + p_val * 0.15
                elif step_name == "db_write":
                    overall = 0.75 + p_val * 0.10
                elif step_name == "embed_chunks":
                    overall = 0.85 + p_val * 0.10
                elif step_name == "vector_write":
                    overall = 0.95 + p_val * 0.04
                elif step_name == "complete":
                    overall = 1.0
                else:
                    overall = p_val

                app_state.indexing_tasks[task_id]["progress"] = round(overall, 3)
                app_state.indexing_tasks[task_id]["current_step"] = step_name
                app_state.indexing_tasks[task_id]["current_message"] = message

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

            if task.get("status") == "aborted":
                add_log("Indexing completed but was flagged as aborted.")
                return

            # ── Auto community-build post-processing ─────────────────────────
            add_log("Starting automatic community-build post-processing...")
            app_state.indexing_tasks[task_id]["current_step"] = "community_build"
            app_state.indexing_tasks[task_id]["current_message"] = "Running community detection..."
            app_state.indexing_tasks[task_id]["progress"] = 1.0

            try:
                from hybrid_rag.graph.community_builder import CommunityBuilder

                community_builder = CommunityBuilder(
                    graph_store=app_state.graph_store,
                    ollama_url=_OLLAMA_URL,
                    llm_model=os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODEL,
                )
                n_communities = await asyncio.to_thread(community_builder.build_communities)
                add_log(f"Community build completed: {n_communities} communities generated.")
            except Exception as community_exc:
                add_log(f"Community build failed (non-fatal): {community_exc}")
                logger.warning("Auto community-build failed for task %s: %s", task_id, community_exc)

            task["status"] = "completed"
            task["completed_at"] = datetime.datetime.now().isoformat()
            add_log(
                f"Indexing completed successfully. Elapsed: {result['elapsed_seconds']:.2f}s. "
                f"Nodes: {result['nodes_upserted']}, Edges: {result['edges_upserted']}, "
                f"Vectors: {result['vectors_upserted']}."
            )

        except Exception as exc:
            if task.get("status") == "aborted" or "Task aborted by user" in str(exc):
                task["status"] = "aborted"
                task["completed_at"] = datetime.datetime.now().isoformat()
                add_log("Indexing aborted by user.")
            else:
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
    translated = translate_path_for_docker(req.repo_path)
    path = Path(translated)
    if not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Provided repo_path does not exist or is not a directory: {req.repo_path}",
        )
    req.repo_path = translated

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
        "progress": 0.0,
        "current_step": "init",
        "current_message": "Task queued.",
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


@app.post("/graph/index/tasks/{task_id}/abort")
async def abort_indexing_task(task_id: str) -> dict[str, str]:
    """Abort a pending or running indexing task."""
    if task_id not in app.state.indexing_tasks:
        raise HTTPException(status_code=404, detail=f"Indexing task not found: {task_id}")

    task = app.state.indexing_tasks[task_id]
    if task["status"] in ("pending", "running"):
        task["status"] = "aborted"
        task["completed_at"] = datetime.datetime.now().isoformat()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task["logs"].append(f"[{timestamp}] Task abort requested by user.")
        logger.info(f"Task {task_id} aborted by user.")
        return {"task_id": task_id, "status": "aborted"}

    return {
        "task_id": task_id,
        "status": task["status"],
        "message": "Task is not in a cancellable/abortable state.",
    }
