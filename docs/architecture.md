# System Architecture & Data Flow

Version: 0.1

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL ENVIRONMENT ONLY                        │
│                 (no data leaves this boundary)                   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │   Codebase   │    │         INDEXING PIPELINE            │   │
│  │  (Python /   │───►│                                      │   │
│  │   Java repo) │    │  1. Language Detection               │   │
│  └──────────────┘    │  2. AST Parsing (tree-sitter)        │   │
│                      │  3. Triplet Extraction               │   │
│                      │  4. Chunking (CodeSplitter)          │   │
│                      │  5. Embedding (nomic-embed-text)     │   │
│                      └──────────┬──────────────────────┬───┘   │
│                                 │                      │        │
│                          Graph Store            Vector Store     │
│                        ┌────────▼──────┐      ┌───────▼─────┐  │
│                        │   FalkorDB    │      │   Qdrant    │  │
│                        │  (Cypher KG)  │      │ (embeddings)│  │
│                        └────────┬──────┘      └───────┬─────┘  │
│                                 │                      │        │
│                      ┌──────────▼──────────────────────▼───┐   │
│                      │       HYBRID RETRIEVAL ENGINE        │   │
│                      │                                      │   │
│  User Query ────────►│  1. Query Analysis                   │   │
│                      │  2. Parallel Execution:              │   │
│                      │     a. Cypher Graph Traversal        │   │
│                      │     b. Vector Similarity Search      │   │
│                      │  3. Reciprocal Rank Fusion (RRF)     │   │
│                      │  4. Context Assembly                 │   │
│                      └──────────────────────┬──────────────┘   │
│                                             │                   │
│                      ┌──────────────────────▼──────────────┐   │
│                      │         INFERENCE LAYER              │   │
│                      │   Ollama (qwen2.5-coder:7b / MLX)   │   │
│                      └──────────────────────┬──────────────┘   │
│                                             │                   │
│                                         Answer                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Indexing Pipeline

**Trigger:** Manual CLI command or file watcher (future).

```
Input:  Local repo directory
Output: Populated FalkorDB + Qdrant collections

Steps:
  File Discovery
      │ glob *.py / *.java
      ▼
  Language Detection
      │ file extension + shebang check
      ▼
  AST Parsing (tree-sitter)
      │ produces: functions, classes, imports, call sites
      ▼
  Triplet Extraction
      │ maps AST nodes → KG schema (see docs/schema/kg-schema.md)
      │ produces: (subject, predicate, object) + properties
      ▼
  Entity Resolution
      │ dedup by ID key, merge stubs for external deps
      ▼
  Graph Write (FalkorDB)
      │ batch Cypher CREATE statements
      ▼
  Chunking (CodeSplitter)
      │ AST-aware chunks, preserves function/class boundaries
      │ chunk_size: 512 tokens, overlap: 64 tokens
      ▼
  Embedding (nomic-embed-text via Ollama)
      │ each chunk → 768-dim vector
      ▼
  Vector Write (Qdrant)
      collection: "code_chunks"
      metadata: {file_path, node_type, name, line_start, line_end}
```

**Key constraint:** No network calls during indexing. All embedding via local Ollama.

---

### 2. Hybrid Retrieval Engine

**Trigger:** User query via API.

```
Input:  Natural language query string
Output: Ranked list of code context chunks

Steps:
  Query Analysis
      │ classify: structural | semantic | hybrid
      │ extract: entity names, relationship keywords
      ▼
  ┌───────────────────────────────────────────┐
  │            PARALLEL EXECUTION             │
  │                                           │
  │  Graph Path                Vector Path    │
  │  ──────────────────  ───────────────────  │
  │  Entity extraction   Embed query          │
  │      │               (nomic-embed-text)   │
  │      ▼                      │             │
  │  Cypher query gen            ▼            │
  │      │               Qdrant top-K search  │
  │      ▼               (K=10, cosine sim)   │
  │  FalkorDB traverse           │             │
  │  (max hops: 3)               │             │
  │      │                      │             │
  │  Graph results      Vector results        │
  └────────────┬─────────────────┬────────────┘
               │                 │
               ▼                 ▼
         RRF Merger (k=60, equal weights default)
               │
               ▼
         Top-N context chunks (N=5 default)
               │
               ▼
         Context Assembly (format for LLM prompt)
```

**RRF formula:**
$$score(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}$$

where k=60, R = {graph_results, vector_results}.

---

### 3. Inference Layer

```
Input:  Assembled context + user query
Output: Natural language answer

Prompt structure:
  [SYSTEM]  You are a code analysis assistant. Answer only from provided context.
  [CONTEXT] {assembled code chunks from retrieval}
  [QUERY]   {user query}

Model: qwen2.5-coder:7b (via Ollama REST API at localhost:11434)
Fallback: llama3.2:3b for low-RAM situations
```

**No streaming in v1** — full response wait. Add streaming in API layer (M4 milestone).

---

## Data Contracts (Module Interfaces)

### Ingestion → FalkorDB
```python
# Node
{"id": str, "label": str, "properties": dict}

# Edge
{"src_id": str, "rel": str, "dst_id": str, "properties": dict}
```

### Ingestion → Qdrant
```python
# Point
{
  "id": uuid,
  "vector": List[float],  # 768-dim
  "payload": {
    "text": str,          # raw chunk
    "file_path": str,
    "node_type": str,     # "function" | "class" | "module"
    "name": str,
    "line_start": int,
    "line_end": int
  }
}
```

### Retrieval → LLM
```python
{
  "query": str,
  "context_chunks": List[{
    "text": str,
    "source": str,        # "graph" | "vector" | "both"
    "file_path": str,
    "score": float        # RRF score
  }]
}
```

---

## Technology Stack

| Component | Technology | Version | Port |
|-----------|-----------|---------|------|
| Inference | Ollama + qwen2.5-coder:7b | latest | 11434 |
| Embedding | Ollama + nomic-embed-text | latest | 11434 |
| Orchestration | LlamaIndex | ≥0.10 | — |
| Graph Store | FalkorDB | latest | 6379 |
| Vector Store | Qdrant | latest | 6333 |
| AST Parser | tree-sitter | ≥0.20 | — |
| API Layer | FastAPI | ≥0.110 | 8000 |
| Language | Python | ≥3.11 | — |

---

## Privacy Boundary Enforcement

All data stays local. Enforced by:
1. No API keys in codebase (`.env.example` only, `.env` gitignored)
2. Ollama runs fully offline — no telemetry
3. FalkorDB + Qdrant run in Docker with no external port exposure beyond localhost
4. CI/CD pipeline: lint checks for any `requests` calls to non-localhost URLs

---

## Deployment (Local Dev)

```bash
make up       # docker compose up falkordb + qdrant
make index    # run ingestion pipeline on target repo
make serve    # start FastAPI on :8000
make test     # pytest
make down     # docker compose down
```

---

## Open Questions (resolve before M1 complete)

- [ ] tree-sitter vs Python `ast` module — see experiments/spike-ast-parsing.ipynb
- [ ] Max hop limit: 3 fixed or query-adaptive?
- [ ] RRF weights: equal (0.5/0.5) or graph-weighted for structural queries?
- [ ] Chunk size: 512 tokens optimal? Validate on LlamaIndex repo.
