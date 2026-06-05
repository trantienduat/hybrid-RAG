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
ollama pull qwen2.5-coder:7b
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
