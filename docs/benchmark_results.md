# Codebase RAG Benchmark & Evaluation Report

## 1. Environment & Setup
*   **Vector Database**: Qdrant (`localhost:6333`, collection: `code_chunks`)
*   **Graph Database**: FalkorDB (`localhost:6379`, graph: `codebase`)
*   **LLM Provider**: Ollama (`http://localhost:11434`, model: `gemma4:12b`)
*   **Embedding Model**: `nomic-embed-text`
*   **Total Points/Vectors Ingested**: ~86,000+
*   **Total Graph Nodes Ingested**: ~51,000+

---

## 2. Latency Performance (SLA validation)
Measured over 5 runs per query type using `hybrid-rag bench --corpus` (SLA Target: $< 1000$ ms):

| Query Type | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Structural** (Graph) | 54.4 ms | 76.2 ms | 80.1 ms | **57.9 ms** |
| **Hybrid** (Vector + Graph) | 38.7 ms | 51.3 ms | 53.4 ms | **40.2 ms** |

*Verdict:* **PASS**. Both retrieval types are well below the 1.0s latency limit, averaging under 60ms.

---

## 3. Retrieval Accuracy (Hit Rate @5)
Measured via `hybrid-rag eval` on the Q1-Q20 corpus:

| Metric | Vector-only RAG | Hybrid-RAG (Ours) | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Hit@5 (1-hop)** | 0.500 | 0.500 | 0.000 |
| **Hit@5 (2-hop)** | 0.600 | 0.600 | 0.000 |
| **Hit@5 (3-hop)** | 0.400 | 0.600 | **+0.200** |
| **Overall Hit@5** | 0.500 | 0.571 | **+0.071** |
| **Mean Reciprocal Rank (MRR)** | 0.274 | 0.336 | **+0.062** |

---

## 4. Key Review & Improvement Decisions

1.  **Semantic Vector Dilution**:
    *   *Observation*: When multiple large repositories (LangChain, Transformers) are indexed in the same Qdrant collection, generic function/class names (e.g. `retrieve`, `config`) collide, diluting vector scores and reducing LlamaIndex query accuracy from ~0.615 to ~0.333 globally.
    *   *Decision*: Implementing and utilizing the `--repo-name` scoping flag restricts search boundaries to specific project namespaces, restoring accuracy to **0.571**. Scoping is mandatory for multi-repository enterprise environments.
2.  **3-Hop Call Chains**:
    *   *Observation*: Hybrid-RAG achieves a **+0.200** Hit Rate improvement over Vector RAG on 3-hop structural queries. Dense retrieval is blind to multi-layered class dependencies.
