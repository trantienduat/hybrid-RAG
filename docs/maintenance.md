# Maintenance & Codebase Guide

This document outlines the codebase architecture, services maintenance, testing protocols, and troubleshooting procedures for the Hybrid-RAG platform.

---

## 🏗️ Codebase Directory Structure & Patterns

Hybrid-RAG is structured following the **Ports and Adapters** (Hexagonal) pattern. Concrete vendor technologies are isolated behind abstract interface ports, ensuring the core RAG logic remains decoupled and independent.

```
src/hybrid_rag/
├── ports/                 # Abstract Interface Definitions (Ports)
│   ├── __init__.py
│   ├── embedder.py        # BaseEmbedder port
│   ├── graph_store.py     # GraphStore port
│   └── vector_store.py    # VectorStore port
├── graph/                 # FalkorDB Concrete Adapter
│   ├── __init__.py
│   └── falkordb_store.py
├── vector/                # Qdrant Concrete Adapter
│   ├── __init__.py
│   └── qdrant_store.py
├── ingestion/             # parsing, resolving, chunking logic
│   ├── parser.py          # tree-sitter AST extraction
│   ├── entity_resolver.py # FQN & Global resolver
│   ├── chunker.py         # AST-aware node splitting
│   └── ollama_embedder.py # Ollama embedding batcher
├── retrieval/             # Hybrid RAG Retriever Engine
│   ├── query_analyzer.py  # Query classification & keywords
│   ├── graph_retriever.py # Cypher expansion
│   ├── vector_retriever.py# Vector lookup
│   ├── rrf.py             # Reciprocal Rank Fusion
│   └── context_assembler.py# Token-budget context assembly
├── api/                   # FastAPI Web & Premium Visual Explorer
│   ├── main.py
│   └── static/
└── cli.py                 # Typer Command Line Interface
```

### Decoupling and Technology Swaps
*   To swap the vector database (e.g. from Qdrant to PGVector), create a new adapter in `vector/` that implements the abstract methods of `VectorStore` in `ports/vector_store.py`.
*   To swap the graph database (e.g. from FalkorDB to Neo4j), implement a `Neo4jStore` implementing the `GraphStore` interface in `ports/graph_store.py`.

---

## 🐳 Containerized Stack & Service Management

The local deployment runs as a fully integrated containerized stack:
1.  **`falkordb`:** Knowledge Graph database, running on port `6379` (FalkorDB Browser at port `8008`).
2.  **`qdrant`:** Vector database, running on port `6333` (REST) and `6334` (gRPC).
3.  **`hybrid-rag-api`:** REST API and premium Cyber-Dark Web Explorer, running on port `8000`.
4.  **`hybrid-rag-mcp`:** Model Context Protocol (MCP) Server, running on port `8001`.
5.  **`otel-collector`:** OpenTelemetry Collector Gateway, running on port `4317` (OTLP gRPC) and `4318` (OTLP HTTP).
6.  **`prometheus`:** System metrics aggregator, running on port `9090`.
7.  **`loki`:** Logs aggregation storage, running on port `3100`.
8.  **`tempo`:** Distributed traces storage, running on port `3200`.
9.  **`grafana`:** Unified telemetry dashboard, running on port `3010` (credentials: `admin` / `admin`).
10. **`phoenix`:** Specialized LLM prompt execution and trace evaluator, running on port `6006`.

### Common Docker Operations

*   **Start/Build Entire Stack:** `docker compose up --build -d`
*   **Stop Entire Stack:** `docker compose down`
*   **Check Live Container Logs:** `docker compose logs -f`
*   **Service Status & Health Check:** `docker compose ps`
*   **Hard Reset & Clear All Data:**
    ```bash
    # Stops all containers and deletes persistent database volumes
    docker compose down -v
    ```

---

## 🧪 Testing Suite

Automated testing is critical to maintain codebase integrity.

### Running Tests
Execute the entire unit and integration test suite:
```bash
.venv/bin/pytest tests/ -v
```

### Running Specific Tests
To run only the Multi-Repository integration tests:
```bash
.venv/bin/pytest tests/integration/test_cross_repo.py -v
```

---

## 🛠️ Troubleshooting & Tuning

### 1. Ollama Connection or Embedding Timeouts
*   **Symptom:** Ingestion hangs or crashes during the "Embedding chunks" stage.
*   **Cause:** Ollama is overloaded, sleeping, or your local machine lacks CPU/GPU RAM.
*   **Fix:**
    *   Ensure Ollama is active (`ollama list`).
    *   Verify the embedding model is loaded: `ollama run nomic-embed-text`.
    *   Tune concurrency in [ollama_embedder.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/ollama_embedder.py) by setting the environment variable `EMBED_CONCURRENCY`. Lower it to `2` or `4` on lower-end hardware (default: `8`).

### 2. FalkorDB Cypher Queries Return Empty
*   **Symptom:** Structural queries return `graph=False` or empty seed lists.
*   **Cause:** FQN ID mismatch or missing namespace.
*   **Fix:**
    *   Ensure FQN IDs match. You can query FalkorDB directly using `redis-cli`:
        ```bash
        redis-cli
        # Select graph and list node IDs
        GRAPH.QUERY codebase "MATCH (n) RETURN n.id, n.repository LIMIT 20"
        ```
    *   Verify that you passed the correct `--repo-name` when indexing. If you query using `--repo-name core-lib`, only nodes with `repository: 'core-lib'` are evaluated as seeds.

### 3. Qdrant Payload Filter Fails
*   **Symptom:** Scoped queries return vector chunks from other repositories.
*   **Cause:** Payload formatting discrepancy or Qdrant collection mismatch.
*   **Fix:**
    *   Check payload keys. Every point in the Qdrant database must have a `repository` key in its payload.
    *   Access the Qdrant REST API to debug collections:
        ```bash
        curl http://localhost:6333/collections/code_chunks
        ```
