# hybrid-RAG

Privacy-preserving Graph-Hybrid RAG for relationship-aware codebase understanding.

## Quickstart

### 1. Prerequisites
- **Python**: 3.12 or higher.
- **Docker**: For running FalkorDB (Graph) and Qdrant (Vector).
- **Ollama**: For local embeddings and LLM extraction.
  - Pull necessary models:
    ```bash
    ollama pull nomic-embed-text
    ollama pull qwen2.5-coder:7b
    ```

### 2. Setup
Install dependencies and initialize the environment:
```bash
make install
```

### 3. Infrastructure
Start the database services:
```bash
make up
```
Check health of services:
```bash
make status
```

### 4. Usage

#### Indexing a Codebase
To index a repository (e.g., the included `llama_index_core` fixture):
```bash
make index REPO=./fixtures/llama_index_core/llama-index-core
```

#### Querying
Ask questions about the indexed codebase:
```bash
make query Q="How does the base response synthesizer work?"
```
```bash
make query Q="Identify all classes that inherit from BaseRetriever and list their specific implementations of the _retrieve method."
```
```bash
make query Q="Which classes in llama-index-core implement the QueryComponent interface and how are they linked to the Pipeline module?"
```
```bash
make query Q="Trace the inheritance hierarchy of PropertyGraphIndex. Does it share any common base classes with VectorStoreIndex?"
```

## Makefile Commands
- `make up`: Start FalkorDB and Qdrant.
- `make down`: Stop services.
- `make install`: Install Python dependencies.
- `make status`: Check if services (FalkorDB, Qdrant, Ollama) are reachable.
- `make index REPO=...`: Index a source code directory.
- `make query Q=...`: Perform a hybrid retrieval query.
- `make test`: Run the test suite.
