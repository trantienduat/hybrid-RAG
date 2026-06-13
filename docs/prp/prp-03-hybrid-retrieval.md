# Product Requirement Prompt (PRP) — Hybrid Retrieval & RRF Fusion

## 🎯 Role & Objective
You are an expert AI software engineer. Your task is to implement the **Hybrid Retrieval Engine** for the `hybrid-RAG` codebase understanding platform. The engine must accept a natural language question, analyze its intent, perform parallel graph retrieval (via FalkorDB) and semantic vector search (via Qdrant), merge the ranked candidates using Reciprocal Rank Fusion (RRF), and assemble a formatted context payload respecting token/character limits.

---

## 🏛️ Architecture & Tech Stack
*   **Architecture:** Hexagonal. The retriever orchestrates calls to the `BaseGraphStore` and `BaseVectorStore` ports.
*   **Language:** Python >=3.12.
*   **Libraries:** `numpy` (optional, for scoring), custom parser/RRF modules.

---

## 🛠️ Functional Requirements

### 1. Query Analyzer ([retrieval/query_analyzer.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/retrieval/query_analyzer.py))
*   Classify the query into one of three types:
    *   `structural`: Triggered by structural words (e.g., "inherit", "subclass", "calls", "implements", "imports") or specific code symbols.
    *   `semantic`: Triggered by conceptual questions (e.g., "explain", "how to", "architecture").
    *   `hybrid`: Mixed queries.
*   Extract code entities (CamelCase class names, snake_case function names, quoted text) and general keyword search terms.

### 2. Graph Retriever ([retrieval/graph_retriever.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/retrieval/graph_retriever.py))
*   Given extracted entities and query type:
    *   Execute Cypher queries against `GraphStore` to find matching class, function, or module nodes.
    *   If query type is `structural`, recursively fetch neighbors (1-2 hops) to capture callers, inheritance chains, and definition paths.
    *   Assign a structural rank or search score to retrieved entities.

### 3. Vector Retriever ([retrieval/vector_retriever.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/retrieval/vector_retriever.py))
*   Embed the user's natural language question using the `BaseEmbedder`.
*   Execute a vector similarity search on Qdrant, retrieving top candidates. Restrict search hits by repository namespace if a filter is provided.

### 4. Reciprocal Rank Fusion Merger ([retrieval/rrf.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/retrieval/rrf.py))
*   Merge structural and semantic candidate lists using the RRF algorithm:
    $$RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$
    *   Constant $k$ defaults to `60`.
    *   Configure custom weights: Graph weight $w_{graph} = 3.0$ and Vector weight $w_{vector} = 1.5$ (structural queries favor graph, semantic favors vector).
*   Merge chunk text and metadata from corresponding candidates, returning a single deduplicated list sorted by RRF score.

### 5. Context Assembler ([retrieval/context_assembler.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/retrieval/context_assembler.py))
*   Assemble the final prompt context by collecting candidate text snippets.
*   Enforce a token or character budget (e.g. `max_tokens`, `max_chars`) to prevent exceeding LLM context windows. Stop adding candidates once the budget is exceeded.
*   Format output as clean Markdown:
    ```markdown
    [File: path/to/file.py]
    Class: ClassName
    ```code...```
    ```

---

## 📈 Non-Functional Requirements & Tests
*   **Offline Isolation:** Ensure retrieval tests can run without real FalkorDB or Qdrant services using mocked stores.
*   **Performance:** Parallelize graph and vector search calls to minimize latency.
