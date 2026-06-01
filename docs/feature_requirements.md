# Feature & System Requirements

This document defines the functional and non-functional requirements implemented in the Hybrid-RAG platform.

---

## 📋 Functional Requirements

### 1. AST Multi-Language Parsing
*   **Req 1.1:** The system must parse both **Python** and **Java** source code.
*   **Req 1.2:** The parser must extract structural entities: **Modules**, **Classes**, and **Functions**.
*   **Req 1.3:** The parser must extract AST-based relationships:
    *   `DEFINES` (e.g. Module defines Class/Function)
    *   `INHERITS` (e.g. Class inherits from Class)
    *   `CALLS` (e.g. Function calls Function)
    *   `IMPORTS` (e.g. Module imports Module)

### 2. Fully Qualified Names (FQN) Mapping
*   **Req 2.1:** All structural graph nodes must use dotted-notation Fully Qualified Names (`package.module.Class.method`) as their unique database identifiers.
*   **Req 2.2:** Node IDs must be dynamically resolved relative to the indexed repository root folder.
*   **Req 2.3:** Node names must be isolated by repository namespace properties to prevent entity collisions in large-scale multi-repo index deployments.

### 3. Parallel & Batch Embedding Ingestion
*   **Req 3.1:** The ingestion pipeline must support chunking source code according to AST boundaries (e.g. chunking per Function/Class).
*   **Req 3.2:** Text chunks must be submitted in parallel batches (default batch size: `128`, default thread concurrency: `8`) to local Ollama.
*   **Req 3.3:** The pipeline must handle slow model responses or rate-limits through automatic retry loops.

### 4. Cross-Repository Knowledge Graph (Multi-Repo RAG)
*   **Req 4.1:** The system must allow partitioning nodes (FalkorDB) and vector points (Qdrant) using a unique `repository` property.
*   **Req 4.2:** During ingestion, the system must execute **Global Entity Resolution** against FalkorDB.
*   **Req 4.3:** If an external stub inherits or imports a class/module already indexed in another repository namespace, the resolver must drop the stub and rewrite graph relationship edges to link directly to the remote repository FQN node.

### 5. Scoped Hybrid Retrieval
*   **Req 5.1:** The retrieval engine must support running scoped searches by accepting a `repository` namespace parameter.
*   **Req 5.2:** Scoped searches must filter Qdrant payloads to ensure only vectors belonging to the target repository are matched.
*   **Req 5.3:** Scoped searches must filter FalkorDB seed node queries to ensure graph path expansions originate only from the target repository.
*   **Req 5.4:** The retrieval engine must support running global, cross-repository retrieval queries when no scope is specified.

### 6. Reciprocal Rank Fusion (RRF) Fusing
*   **Req 6.1:** Semantic vector candidates and structural graph path candidates must be fused using the Reciprocal Rank Fusion algorithm.
*   **Req 6.2:** The fusion algorithm must apply custom rank-bias weights:
    *   `W_graph = 3.0` for structural questions.
    *   `W_graph = 1.5` for hybrid questions.

### 7. Token-Budget Context Assembly
*   **Req 7.1:** The prompt context assembler must support limiting prompt sizes using strict token-budget limits (`--max-tokens`) or character-budget limits (`--max-chars`).
*   **Req 7.2:** Chunks must be packed dynamically in descending order of their RRF rank until the budget limit is met.
*   **Req 7.3:** Detailed assembly statistics must be returned in query metadata (number of chunks included, number of chunks excluded, total tokens/characters utilized).

### 8. Web Visualization & API Streaming
*   **Req 8.1:** The server must host a premium web-based Knowledge Graph explorer supporting dynamic repository selection, 3D Force-Graph visualization, a Cytoscape 2D hierarchical layout view switcher, search category tabs, and an interactive Floating Node Inspector panel.
*   **Req 8.2:** The API must support Server-Sent Events (SSE) streaming (`/query/stream`) to stream LLM responses chunk-by-chunk for low-latency user interfaces.

---

## ⚡ Non-Functional Requirements

### 1. Data Privacy & Sovereignty
*   **Req 1.4:** No code, embeddings, queries, or LLM contexts may leave the local boundary. External network requests are prohibited.
*   **Req 1.5:** Databases and model servers must run offline within standard localhost Docker and process boundaries.

### 2. High-Performance Indexing and Querying
*   **Req 2.4:** Sub-second retrieval latency: Retrieval fusion (vector + graph) must execute in **less than 1 second** on standard local CPU/GPU hardware.
*   **Req 2.5:** Parallel indexing must speed up Nomics/Ollama embedding workflows by at least **5x** compared to sequential indexing.

### 3. Containerized Deployment & Portability
*   **Req 3.1:** The entire RAG stack (API, Web Visualizer, Graph DB, and Vector DB) must be deployable with a single command (`docker compose up --build -d`) to ensure fast, isolated, compiler-free installation on private/on-premise machines.
