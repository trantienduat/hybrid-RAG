# Hybrid-RAG answer-quality benchmark comparison

Run date: 2026-08-08
Evidence grade: `approved_ai_source_review`

## Protocol

Both repositories used the same benchmark protocol: 10 source-anchored gold cases, 3 repeats, 30 hybrid answers, 30 vector-only answers, `top_k=20`, `context_n=5`, and seed `42`. Answers used `gemma4:12b`; RAGAS judge metrics used the independent `qwen2.5-coder:7b`; embeddings used `nomic-embed-text`.

| Workload | Pinned source | Graph | Qdrant collection | Vector points | Index run |
| --- | --- | --- | --- | ---: | --- |
| Transformers v5.9.0 | `huggingface/transformers@0a2757da521a7a49b8143d9e0c938f08747d682e` | `bench_transformers_20260807` | `transformers_bench_20260807` | 25 | `ea222cfc-4015-4f74-86da-42d370e2d526` |
| LangChain Core 1.4.7 | `langchain-ai/langchain@51578289bb1f696a643e0740be1441039d8af8ce` (`libs/core`) | `bench_langchain_core_20260807` | `langchain_core_bench_20260807` | 20 | `bde4837e-8e59-4e35-a272-ec8011ec2d5d` |

The graph indexes contain the parsed five-file source slices used by each gold manifest. The vector collections are curated anchor-derived slices from those files, so this is a reproducible source-backed comparison, not a claim about full-repository retrieval performance.

## Results

Scores are means over 30 answers per mode. Latencies are milliseconds; token counts are averages.

| Workload | Mode | Source hit | Faithfulness | Relevancy | Correctness | Retrieval ms | Generation ms | Prompt tokens | Completion tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformers | Hybrid | 1.000 | 0.967 | 0.918 | 0.759 | 347.3 | 90,270.4 | 1,387.0 | 746.0 |
| Transformers | Vector | 1.000 | 1.000 | 0.895 | 0.718 | 122.4 | 97,336.4 | 2,261.1 | 763.4 |
| LangChain Core | Hybrid | 1.000 | 0.867 | 0.712 | 0.581 | 288.4 | 61,803.5 | 1,218.3 | 477.1 |
| LangChain Core | Vector | 1.000 | 0.967 | 0.803 | 0.636 | 123.5 | 70,137.5 | 1,550.6 | 561.9 |

### Hybrid minus vector deltas

| Workload | Faithfulness | Relevancy | Correctness | Retrieval ms | Generation ms | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformers | -0.033 | +0.023 | +0.041 | +225.0 | -7,066.0 | -874.1 |
| LangChain Core | -0.100 | -0.091 | -0.055 | +164.9 | -8,334.0 | -332.3 |

## Reading the result

- Transformers favors hybrid retrieval on answer relevancy and correctness, while using less context and reducing generation time; its graph path adds retrieval overhead and slightly lowers faithfulness.
- LangChain Core favors vector-only retrieval on all three judge metrics in this slice. Hybrid still reduces prompt size and generation time, but adds retrieval latency.
- Source-hit rate is saturated at 1.0 for both modes and both workloads, so the meaningful separation here is answer quality and efficiency rather than whether the gold file was retrieved.

## Limitations and reproducibility notes

- Gold answers were reviewed by AI against pinned source anchors; no independent human adjudication was performed.
- The vector corpus is the small, source-anchored benchmark slice described above rather than a full-repository index.
- Generation and judging ran locally through Ollama; estimated API costs are unavailable (`null`).
- LangChain repeats with identical question, answer, reference, and contexts were scored once and memoized across exact duplicates. This is valid for the configured deterministic judge (`temperature=0`) and leaves the 30-record-per-mode aggregates unchanged.

Canonical raw reports:

- `results/transformers_v5_9_answer_quality_benchmark.json`
- `results/langchain_core_v1_4_7_answer_quality_benchmark.json`
