# Hybrid-RAG Project Operating System

## Mission

Help developers understand large, private codebases by combining structural knowledge-graph traversal with semantic retrieval, while keeping the default workflow local, measurable, and practical to operate.

## Current Focus

- Turn the implemented Hybrid-RAG platform into a reproducible, thesis-ready system rather than adding broad new capabilities.
- Validate retrieval and answer quality with trustworthy, versioned evaluation runs.
- Close the remaining scale, graph-fidelity, and operational gaps on representative large repositories.
- Keep documentation, acceptance criteria, and generated evidence aligned with the behavior of the current code.

## Current Milestone

**M4 — API, Visualization, and Evaluation: evidence closure**

The API, streaming UI, MCP interface, retrieval benchmark, and RAGAS pipeline are implemented. The milestone remains open until the scale test, final RAGAS results, and thesis-ready results package are complete and reproducible from a clean environment.

## Success Criteria

- A valid evaluation run has 100% ground-truth coverage and matching graph/vector provenance.
- Hybrid retrieval achieves Hit Rate @5 of at least 0.90 for 1-hop queries and 0.75 for 2–3-hop queries.
- Hybrid retrieval beats vector-only retrieval by at least +0.20 Hit Rate on 2–3-hop queries.
- Overall MRR is at least 0.70.
- RAGAS faithfulness is at least 0.80 and answer relevance is at least 0.75, with failures reported rather than silently converted into scores.
- Structural retrieval latency is below 1 second at p95 on documented hardware and dataset versions.
- A representative large-repository scale test completes with recorded runtime, resource usage, corpus provenance, and retrieval results.
- The API, web UI, CLI, and MCP paths use the same retrieval semantics and repository-scoping rules.
- `make lint` and `make test` pass; live integration results are recorded when FalkorDB, Qdrant, and Ollama are available.

## Capability Health

| Capability | Health | Evidence / gap |
|---|---|---|
| Python ingestion and AST graph extraction | Healthy | Implemented with unit and integration coverage; used by the current evaluation flow. |
| Hybrid graph + vector retrieval | Healthy | Latest valid 50-query run reports Hit@5 of 1.000 with 50/50 ground-truth coverage. |
| Repository scoping and multi-repo retrieval | Healthy | Implemented across graph and vector paths; provenance validation is part of evaluation. |
| Incremental and concurrent indexing | Healthy | Implemented with dedicated tests and architecture decisions. |
| REST API, SSE web UI, CLI, and MCP | Healthy | Implemented; consistency across entry points remains an ongoing regression concern. |
| Retrieval observability and latency | Watch | OpenTelemetry and benchmarks exist; detailed per-stage latency profiling is still backlog work. |
| Answer-quality evaluation | Watch | RAGAS pipeline exists and scoring was recently hardened; current accepted faithfulness and relevance results still need to be published. |
| Large-repository validation | At risk | LlamaIndex evaluation exists, but the roadmap's Transformers scale test is not complete. |
| Java structural extraction | At risk | Java is accepted by configuration, but the README states Java AST-to-graph extraction is not implemented. |
| Reproducible local deployment | Watch | Docker Compose path exists; full live validation depends on FalkorDB, Qdrant, and Ollama services and host model availability. |

## Top Risks

1. **Benchmark confidence exceeds product confidence.** The diagnostic ground truth is derived from the indexed graph, so a perfect retrieval score does not prove that the graph completely represents the source code.
2. **Answer-quality evidence is incomplete.** Retrieval can pass while generated answers remain unfaithful or irrelevant; accepted RAGAS results are still required.
3. **Large-repository behavior is under-validated.** Resource use, indexing duration, and retrieval quality may regress at Transformers-scale corpora.
4. **Documentation can drift from implementation.** Roadmap, requirements, README limitations, and generated reports currently describe different slices of project status.
5. **Local infrastructure affects reproducibility.** Model versions, hardware, and live data services can materially change latency and quality results.
6. **Declared language scope is wider than delivered scope.** Configuration and requirements mention Java, while production extraction remains Python-first.

## Working Agreements

- Use `develop` as the integration base; keep changes focused and reviewable on short-lived branches.
- Define acceptance evidence before implementation and update it when behavior or datasets change.
- Treat generated evaluation output as valid only when coverage, provenance, configuration, dataset revision, and hardware are recorded.
- Never present partial, stale, or error-derived evaluation results as passing.
- Add or update tests with behavior changes; run `make lint` and `make test` before merge.
- Mark tests that require FalkorDB, Qdrant, Ollama, or cloud providers explicitly and record when they were not run.
- Write an ADR for decisions that change architecture, storage, retrieval semantics, data boundaries, or operational constraints.
- Keep the default path local and privacy-preserving; make external provider usage explicit.
- Update this file when the current focus, milestone, health, or top risks materially change.
