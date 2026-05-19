# Roadmap

Reflects thesis milestones. Updated as work progresses.

---

## Current State — May 2026

| Item | Status |
|------|--------|
| Smoke test (FalkorDB, Qdrant, Ollama) | ✅ Done |
| 20 multi-hop test queries | ✅ Done |
| KG schema v0.1 | ✅ Done |
| Architecture data flow | ✅ Done |
| Project structure init | ✅ Done |
| `pyproject.toml` / `docker-compose.yml` / `Makefile` | ✅ Done |
| First line of src code | ✅ Done |

---

## M1 — Ingestion Pipeline `Weeks 1–2`

**Goal:** Given a local Python repo, produce populated FalkorDB + Qdrant.

| Task | Issue | Status |
|------|-------|--------|
| Spike: tree-sitter vs ast module | #1 | ✅ |
| ADR: AST parser choice | #2 | ✅ |
| `ingestion/parser.py` — AST → raw nodes | #3 | ✅ |
| `ingestion/triplet_extractor.py` — raw nodes → KG triplets | #4 | ✅ |
| `ingestion/entity_resolver.py` — dedup + merge | #5 | ✅ |
| `graph/client.py` — FalkorDB write client (`FalkorDBStore` adapter) | #6 | ✅ |
| `ingestion/chunker.py` — AST-aware code splitting | #7 | ✅ |
| `vector/client.py` — Qdrant write client (`QdrantStore` adapter) | #8 | ✅ |
| `ingestion/embedder.py` — local embedding via Ollama | #9 | ✅ |
| `cli.py` — `hybrid-rag index <repo>` command | #10 | ✅ |
| Unit tests for parser + extractor | #11 | ✅ |
| Integration test: index small fixture repo end-to-end | #12 | ✅ |

**M1 Done When:** `make index REPO=./fixtures/small_repo` runs without error, FalkorDB has nodes/edges, Qdrant has vectors.

---

## M2 — Auto-KGC + Entity Resolution `Weeks 3–5`

**Goal:** LLM-assisted triplet extraction for relationships AST cannot capture (semantic imports, dynamic calls).

| Task | Issue | Status |
|------|-------|--------|
| Spike: LLM triplet extraction prompt design | #13 | ✅ |
| `ports/llm_extractor.py` (ABC) + `ingestion/ollama_llm_extractor.py` (adapter) | #14 | ✅ |
| Merge LLM results with AST results | #15 | ✅ |
| Improve entity resolution: cross-file class linking | #16 | ✅ |
| Handle external/stdlib stubs | #17 | ✅ (done in M1 via `entity_resolver.py`) |
| Test: index LlamaIndex repo (medium-scale) | #18 | ⬜ |
| Graph fidelity evaluation vs known structure | #19 | ⬜ |

**M2 Done When:** Graph on LlamaIndex repo passes Q1–Q10 from evaluation.md with correct Cypher results.

---

## M3 — Hybrid Retrieval + RRF `Weeks 6–8`

**Goal:** Parallel graph + vector retrieval fused via RRF, outperforms vector-only on multi-hop queries.

| Task | Issue | Status |
|------|-------|--------|
| `retrieval/query_analyzer.py` — classify structural/semantic/hybrid | #20 | ✅ |
| `retrieval/graph_retriever.py` — Cypher query generator + executor | #21 | ✅ |
| `retrieval/vector_retriever.py` — Qdrant semantic search | #22 | ✅ |
| `retrieval/rrf.py` — Reciprocal Rank Fusion merger | #23 | ✅ |
| `retrieval/context_assembler.py` — format context for LLM | #24 | ✅ |
| Baseline evaluation: Vector-only vs Hybrid on Q1–Q20 | #25 | ✅ |
| Tune RRF k + weights | #26 | ✅ |

**M3 Done When:** Hybrid outperforms vector-only by ΔHitRate ≥ +0.20 on 2-3 hop queries (Q11–Q15).

**M3 Results (May 2026):** Winning config: `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`
- Hit@5 all: Hybrid 0.615 vs Vector 0.308 (+0.308)
- Hit@5 3-hop: Hybrid 0.800 vs Vector 0.400 (+0.400)
- ΔHitRate 2-3hop = **+0.222** (target ≥ +0.20 ✓)

---

## M4 — API + Visualization + Evaluation `Weeks 9–12`

**Goal:** Usable system with web interface, full RAGAS evaluation, thesis results.

| Task | Issue | Status |
|------|-------|--------|
| `api/main.py` — FastAPI app with `/query` endpoint | #27 | ✅ |
| `api/schemas.py` — request/response Pydantic models | #28 | ✅ |
| Streaming response support (`/query/stream` SSE) | #29 | ✅ |
| Web visualization: D3.js force-directed graph explorer | #30 | ✅ |
| RAGAS evaluation pipeline on Q1–Q20 | #31 | ✅ |
| Latency benchmarks (p50/p95/p99 per query type) | #32 | ✅ |
| Scale test: Transformers (HuggingFace) repo | #33 | ⬜ |
| Final thesis results write-up data | #34 | ⬜ |

**M4 Done When:** All acceptance criteria in evaluation.md met, API running, results exportable.

**M4 Achieved (commit `feat/m4`):**
- FastAPI REST API with `/query`, `/query/stream` (SSE), `/graph/neighbors/{id}`, `/graph/search`, `/health`
- D3.js force-directed graph explorer + streaming query panel served at `GET /`
- `hybrid-rag serve` CLI command (wraps uvicorn)
- `hybrid-rag ragas` CLI command — RAGAS faithfulness/answer_relevancy/context_precision on Q1–Q20
- `hybrid-rag bench` CLI command — p50/p95/p99 latency per query type (structural/hybrid/semantic)
- Winning RRF config wired into API defaults: `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`

---

## Branching Strategy

```
main          ← stable, tagged releases only
  └─ develop  ← integration branch (default PR target)
       └─ feat/#{issue}-short-desc   ← one branch per issue
       └─ fix/#{issue}-short-desc
       └─ spike/#{issue}-short-desc  ← experiment branches (notebooks)
```

PR checklist:
- [ ] `make lint` passes
- [ ] `make test` passes (or new tests added)
- [ ] Linked to issue number
- [ ] ADR written if architecture decision made

---

## ADR Index

| ID | Decision | Status |
|----|----------|--------|
| 001 | Graph store: FalkorDB | Accepted |
| 002 | AST parser: tree-sitter | Accepted |
| 003 | Embedding model: nomic-embed-text | Accepted |
| 004 | LLM: qwen2.5-coder:7b | Accepted |
