"""
API schemas — Pydantic request/response models for the hybrid-rag FastAPI service.

M4 #28.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /query body."""

    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question.")
    top_k: int = Field(20, ge=1, le=100, description="Retrieval candidates before context assembly.")
    context_n: int = Field(5, ge=1, le=20, description="Top results assembled for LLM context.")
    max_tokens: int | None = Field(None, ge=1, le=16384, description="Maximum tokens for dynamic context budget.")
    max_chars: int | None = Field(None, ge=1, le=65536, description="Maximum characters for dynamic context budget.")
    llm_model: str = Field("qwen2.5-coder:7b", description="Ollama model name for generation.")
    stream: bool = Field(False, description="Set True to use SSE streaming endpoint instead.")
    repository: str | None = Field(None, description="Optional repository name to filter search results and context by.")


# ── Source chunk ───────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """One retrieved code chunk or graph node included in the context."""

    node_id: str
    name: str
    label: str            # Module | Class | Function | Variable
    file_path: str
    text: str             # code snippet text (empty for graph-only nodes)
    source: str           # "graph" | "vector" | "hybrid"
    rrf_score: float


# ── Response ───────────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """POST /query response."""

    question: str
    answer: str
    query_type: str       # structural | semantic | hybrid | global
    sources: list[SourceChunk]
    latency_ms: float


class StreamToken(BaseModel):
    """One SSE event payload for POST /query/stream."""

    token: str
    done: bool


# ── Graph exploration ──────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    name: str
    label: str
    file_path: str
    repository: str = ""


class GraphEdge(BaseModel):
    src_id: str
    rel: str
    dst_id: str
    dst_name: str
    dst_label: str
    dst_file_path: str = ""
    dst_repository: str = ""


class GraphNeighborsResponse(BaseModel):
    """GET /graph/neighbors/{node_id} response."""

    node_id: str
    label: str
    name: str
    neighbors: list[GraphEdge]


class GraphSearchResponse(BaseModel):
    """GET /graph/search response."""

    query: str
    nodes: list[GraphNode]


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health response."""

    status: str           # "ok" | "degraded"
    falkordb: str
    qdrant: str
    ollama: str
