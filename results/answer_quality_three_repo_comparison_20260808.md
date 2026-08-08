# Hybrid-RAG answer-quality benchmark: three-repository comparison

Run date: 2026-08-08
Evidence grade: `approved_ai_source_review`

## Workloads and protocol

All workloads use the repository benchmark protocol with `gemma4:12b` for answers, independent `qwen2.5-coder:7b` RAGAS judging, `nomic-embed-text` embeddings, `top_k=20`, `context_n=5`, three repeats, and seed `42`.

| Workload | Pinned source | Graph | Qdrant collection | Points | Index run |
| --- | --- | --- | --- | ---: | --- |
| LlamaIndex Core 0.14.21 | wheel SHA-256 `4a807d31e54d066068e076eb4d066efbf95e2d2a00dcbe0eba3d9340a04cad42` | `llama_core_answer_v3_20260801` | `llama_core_answer_v3_20260801` | 24,174 | `9ca6e68a-2ffb-4eae-b0de-38d6edd98af5` |
| Transformers v5.9.0 | `huggingface/transformers@0a2757da521a7a49b8143d9e0c938f08747d682e` | `bench_transformers_20260807` | `transformers_bench_20260807` | 25 | `ea222cfc-4015-4f74-86da-42d370e2d526` |
| LangChain Core 1.4.7 | `langchain-ai/langchain@51578289bb1f696a643e0740be1441039d8af8ce` (`libs/core`) | `bench_langchain_core_20260807` | `langchain_core_bench_20260807` | 20 | `bde4837e-8e59-4e35-a272-ec8011ec2d5d` |

LlamaIndex contains 30 gold cases and 90 answers per mode. Transformers and LangChain contain 10 gold cases and 30 answers per mode. The LlamaIndex vector index is a full indexed wheel snapshot; the other two are reproducible source-anchored benchmark slices, so vector point counts and absolute latency should not be compared as like-for-like scale measurements.

## Results

| Workload | Mode | n | Source hit | Faithfulness | Relevancy | Correctness | Retrieval ms | Generation ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LlamaIndex | Hybrid | 90 | 1.000 | 0.924 | 0.808 | 0.698 | 371.1 | 87,268.2 |
| LlamaIndex | Vector | 90 | 1.000 | 0.896 | 0.729 | 0.652 | 134.0 | 96,754.5 |
| Transformers | Hybrid | 30 | 1.000 | 0.967 | 0.918 | 0.759 | 347.3 | 90,270.4 |
| Transformers | Vector | 30 | 1.000 | 1.000 | 0.895 | 0.718 | 122.4 | 97,336.4 |
| LangChain Core | Hybrid | 30 | 1.000 | 0.867 | 0.712 | 0.581 | 288.4 | 61,803.5 |
| LangChain Core | Vector | 30 | 1.000 | 0.967 | 0.803 | 0.636 | 123.5 | 70,137.5 |

## Hybrid minus vector

| Workload | Faithfulness | Relevancy | Correctness | Retrieval ms | Generation ms | Prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LlamaIndex | +0.029 | +0.079 | +0.046 | +237.1 | -9,486.3 | -101.0 |
| Transformers | -0.033 | +0.023 | +0.041 | +225.0 | -7,066.0 | -874.1 |
| LangChain Core | -0.100 | -0.091 | -0.055 | +164.9 | -8,334.0 | -332.3 |

## Takeaway

Hybrid retrieval improved answer correctness on LlamaIndex and Transformers, and improved relevancy on all but LangChain. It consistently reduced generation time and prompt size, at the cost of roughly 165–237 ms additional retrieval latency in these local runs. LangChain is the exception: vector-only retrieval scored higher on all three quality metrics in its source slice.

All modes achieved a 1.0 source-file hit rate, so answer-quality metrics—not source discovery—provide the useful separation in this benchmark set.

## Limitations

- Gold answers are AI-reviewed against pinned source anchors; no independent human adjudication was performed.
- Corpus scale differs: LlamaIndex is a full wheel snapshot, while Transformers and LangChain use small source-anchored slices.
- Local Ollama inference makes absolute generation latency host-dependent; estimated API costs are unavailable.
- LangChain exact duplicate repeat inputs were scored once and memoized across repeats under the deterministic judge configuration.

Canonical reports:

- `results/llama_index_core_v0_14_21_answer_quality_benchmark.json`
- `results/transformers_v5_9_answer_quality_benchmark.json`
- `results/langchain_core_v1_4_7_answer_quality_benchmark.json`
