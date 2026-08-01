# Hybrid-RAG Operations & Quick Start Guide

This guide describes how to quickly deploy, index, and run the Hybrid-RAG system using Docker. By following this 5-minute Quick Start, you will have the database engines, the visualizer Web UI, and the Model Context Protocol (MCP) server running and connected directly to your IDE.

---

## 🚀 1. Quick Start: Setup, Ingestion & IDE Integration

### Step A: Pull Local Models
Ensure Ollama is running locally on your host machine, and download the default embedding and coding models:
```bash
# Pull the high-performance local embedding model
ollama pull nomic-embed-text

# Pull the default coding LLM
ollama pull gemma4:12b
```

### Step B: Start the Service Stack (Docker)
From the project root directory, spin up the entire service stack:
```bash
docker compose up -d --build
```
*This spins up FalkorDB (graph DB), Qdrant (vector DB), the Visualizer Web UI (port `8000`), and the MCP Server (port `8001`).*

Set `CODEBASES_PATH=/absolute/path/to/your/codebases` before starting Compose
to expose additional host repositories under `/codebases`.

### Step C: Ingest & Index Your Codebases
Since the codebase folders reside on the host filesystem, install the CLI locally on your host machine and run the indexing commands:
```bash
# 1. Quickly setup local environment and active virtual environment
make install
source .venv/bin/activate

# 2. Index this Hybrid-RAG repository itself
hybrid-rag index . --repo-name hybrid-rag

# 3. Index all test fixtures (in dependency order for cross-repo entity resolution)
hybrid-rag index ./fixtures/small_repo --repo-name small-app
hybrid-rag index ./fixtures/dependent_repo --repo-name main-app

# 4. Build modular communities for global architectural search
hybrid-rag community-build
```

### Step D: Connect an MCP client
Add the following stdio server definition to an MCP-compatible client, using
the client's documented configuration location:

```json
{
  "mcpServers": {
    "hybrid-rag": {
      "command": "/absolute/path/to/hybrid-RAG/.venv/bin/hybrid-rag",
      "args": ["mcp", "--transport", "stdio"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```
> [!TIP]
> **Running strictly containerized?**
> If you prefer not to use the host-level virtual environment, you can configure the IDE to communicate with the running Docker container using a stdio bridge:
> ```json
> {
>   "mcpServers": {
>     "hybrid-rag": {
>       "command": "docker",
>       "args": ["exec", "-i", "hybrid-rag-mcp", "hybrid-rag", "mcp", "--transport", "stdio"]
>     }
>   }
> }
> ```

*Once saved, reload the client to activate the codebase tools.*

### Step E: Alternative - Trigger Indexing via REST API (Background Service)

Instead of running the indexing synchronously via the CLI on your host, you can trigger repository indexing asynchronously using the REST API. This is especially useful for remote environments, headless servers, or Git Webhook integrations.

Indexing runs in the background with bounded concurrency. Up to three tasks run
at once by default; set `INDEXING_CONCURRENCY` to change the limit.
The API accepts repositories only beneath configured `repositories` paths,
`INDEX_ROOTS`, or the `/codebases` container mount. Use the CLI for an ad-hoc
local path or add its root explicitly before starting the API.

#### 1. Trigger an Indexing Task
Send a `POST` request to `/graph/index` with the absolute path of the repository:

```bash
curl -X POST http://localhost:8000/graph/index \
     -H "Content-Type: application/json" \
     -d '{
       "repo_path": "/Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/fixtures/small_repo",
       "languages": ["python"],
       "repo_name": "small-app-via-api",
       "llm_extract": false,
       "max_tokens": 512
     }'
```

*Response (`200 OK`):*
```json
{
  "task_id": "8c459fd9-e70a-4712-ba78-43956bf3a71b",
  "status": "pending",
  "repository": "small-app-via-api"
}
```

> [!WARNING]
> If running the API inside Docker (`hybrid-rag-api`), the `repo_path` must be accessible **within the container**. You can map host directories as volumes in `docker-compose.yml` (e.g. mapping `/codebases` on the host to `/codebases` in the container) to index them via the REST API.

#### 2. Monitor Task Status & Logs
You can query the status and real-time logs of the indexing process using the returned `task_id`:

```bash
curl http://localhost:8000/graph/index/tasks/8c459fd9-e70a-4712-ba78-43956bf3a71b
```

*Response (`200 OK`):*
```json
{
  "task_id": "8c459fd9-e70a-4712-ba78-43956bf3a71b",
  "repository": "small-app-via-api",
  "status": "running",
  "created_at": "2026-06-05T21:42:00.123456",
  "completed_at": null,
  "logs": [
    "[2026-06-05 21:42:00] Task initialized and queued.",
    "[2026-06-05 21:42:00] Waiting to acquire indexing lock...",
    "[2026-06-05 21:42:01] Lock acquired. Starting indexing pipeline...",
    "[2026-06-05 21:42:01] [parse] Parsing source files in /app/fixtures/small_repo..."
  ],
  "error": null
}
```

#### 3. List All Tasks
To retrieve the history and statuses of all background indexing tasks:

```bash
curl http://localhost:8000/graph/index/tasks
```

---


## 🌐 2. Accessing the Services & Verification

Once setup and indexing are completed, the following services are fully operational:

*   **Interactive Web UI & 3D/2D Visualizer:** Open `http://localhost:8000/` in your browser. Scope queries by repository, view dependency graphs (with directory-based community borders), and chat in real-time.
*   **Model Context Protocol (MCP) Server:** Access the SSE network endpoint at `http://localhost:8001/sse` (For detailed setup and Mermaid architecture diagram, refer to the [MCP Setup Guide](mcp.md)).
*   **REST API Documentation:** Open `http://localhost:8000/docs` to view the FastAPI Swagger UI.
*   **Qdrant Admin Dashboard:** Visit `http://localhost:6333/dashboard` to inspect vector collections.


---

## 🔄 3. Incremental Indexing & Git Sync

To optimize performance and save resources on large repositories, Hybrid-RAG supports **File-level Incremental Indexing**. Instead of re-indexing the entire repository, the system detects modified, added, renamed, or deleted files, performs a scoped cleanup in both FalkorDB and Qdrant, and indexes only the changes.

### Incremental CLI Flags
When running `hybrid-rag index`, you can pass the following options:
*   `--incremental` / `--no-incremental` (Default: `--incremental`): Toggles incremental indexing. If enabled, compares the repository state against the last indexed commit stored in FalkorDB.
*   `--rebuild`: Forces a full rebuild of the repository, ignoring previously indexed commit state.
*   `--from-commit <hash>`: Overrides auto-detection and compares the current codebase against a specific past commit hash. Highly useful inside CI/CD workflows (e.g., GitHub Actions using target commit ranges).

Example:
```bash
# Force a full rebuild
hybrid-rag index . --rebuild

# Run incremental indexing compared to a specific past commit
hybrid-rag index . --from-commit a1b2c3d4
```

### Mono-repo & Subdirectories Support
For enterprise monorepos or multi-package projects, you can configure repository paths in `config.json` targeting nested directories or packages (e.g., `/path/to/monorepo/libs/core`).
*   **Recursive Git Root Discovery**: The system automatically and recursively searches upward from the configured subdirectory path to find the parent `.git` folder.
*   **Automatic Incremental Sync**: Thanks to this upward search, the background sync worker and CLI can correctly run incremental indexing (`git diff`) on monorepo subdirectories without requiring separate Git initializations for each package.

---


## 🛠️ 4. Additional Operational CLI Commands

When developing or running diagnostic evaluations locally, you can use the active python environment to execute the following commands:

*   **Query the codebase via terminal:**
    ```bash
    hybrid-rag query "Explain how SuperCalculator inherits Calculator"
    ```
*   **Check database status and stats:**
    ```bash
    hybrid-rag status
    ```
*   **Run latency benchmarks (p50/p95/p99):**
    ```bash
    hybrid-rag bench --corpus --repo-name hybrid-rag
    ```
*   **Run RAGAS generation-quality evaluation:**
    ```bash
    hybrid-rag ragas --repo-name hybrid-rag
    ```
*   **Compare Hybrid and vector-only generated answers:**
    ```bash
    python scripts/run_answer_quality_benchmark.py --repeats 3
    ```
