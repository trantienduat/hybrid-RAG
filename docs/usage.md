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
hybrid-rag index ./fixtures/llama_index_core --repo-name llama-core

# 4. Build modular communities for global architectural search
hybrid-rag community-build
```

### Step D: Connect to Antigravity IDE (MCP Integration)
To enable the IDE agent to use the hybrid RAG index, copy and paste the configuration block below into your Gemini Code Assist / Antigravity IDE configuration file:
*   **Path (macOS / Linux):** `~/.gemini/config/mcp_config.json`
*   **Path (Windows):** `C:\Users\[YourUsername]\.gemini\config\mcp_config.json`

```json
{
  "mcpServers": {
    "hybrid-rag": {
      "command": "/Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/.venv/bin/hybrid-rag",
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

*Once saved, reload the IDE window (e.g. `Developer: Reload Window` in VS Code) to activate the 5 new codebase tools in your chat agent.*

### Step E: Alternative - Trigger Indexing via REST API (Background Service)

Instead of running the indexing synchronously via the CLI on your host, you can trigger repository indexing asynchronously using the REST API. This is especially useful for remote environments, headless servers, or Git Webhook integrations.

Indexing tasks are queued in a **serialized FIFO queue** inside the running FastAPI container, ensuring that only one repository is processed at a time. This prevents database write locks in FalkorDB and local GPU/VRAM memory exhaustion in Ollama.

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

*   **Interactive Web UI & 3D/2D Visualizer:** Open `http://localhost:8000/` in your browser. Scope queries by repository, view dependency graphs (with Louvain community borders), and chat in real-time.
*   **Model Context Protocol (MCP) Server:** Access the SSE network endpoint at `http://localhost:8001/sse` (For detailed setup and Mermaid architecture diagram, refer to the [MCP Setup Guide](mcp.md)).
*   **REST API Documentation:** Open `http://localhost:8000/docs` to view the FastAPI Swagger UI.
*   **Qdrant Admin Dashboard:** Visit `http://localhost:6333/dashboard` to inspect vector collections.

---

## 🛠️ 3. Additional Operational CLI Commands

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
    hybrid-rag bench --corpus
    ```
*   **Run RAGAS generation-quality evaluation:**
    ```bash
    hybrid-rag ragas
    ```
