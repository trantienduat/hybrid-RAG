# System Usage & Operations Guide

This guide provides instructions on how to set up, operate, and query the Hybrid-RAG system.

---

## ⚙️ Prerequisites & Service Setup

Hybrid-RAG runs entirely locally and requires Docker and Ollama.

### 1. Start Local Databases (Docker)
Ensure Docker is running, then spin up the backend databases (FalkorDB and Qdrant):
```bash
make up
```
*This starts FalkorDB on port `6379` and Qdrant on port `6333` on `localhost`.*

### 2. Set Up Local Models (Ollama)
Ensure Ollama is running locally, then pull the necessary models:
```bash
# Pull the high-performance embedding model
ollama pull nomic-embed-text

# Pull the default coding LLM
ollama pull qwen2.5-coder:7b

# (Optional) Pull low-RAM fallback LLM
ollama pull llama3.2:3b
```

### 3. Setup Python Virtual Environment
Install system dependencies and packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 📦 Ingestion: Indexing Repositories

Use the `hybrid-rag index` command to parse and ingest repositories into FalkorDB and Qdrant.

```bash
hybrid-rag index <PATH_TO_REPO> [OPTIONS]
```

### Key Options
*   `--repo-name <NAME>`: Custom namespace for the repository. Defaults to the folder name. **(Highly recommended for Multi-Repo RAG)**.
*   `--languages <LANG>`: Source languages to extract (default: `python`). Supports `python` and `java`.
*   `--max-tokens <INT>`: Maximum tokens per chunk (default: `512`).
*   `--llm-extract / --no-llm-extract`: Enable/disable LLM-assisted structural edge extraction (default: `--no-llm-extract`).

### Examples

**Index a single library:**
```bash
hybrid-rag index ./fixtures/small_repo --repo-name core-lib
```

**Index a dependent application and trigger cross-repo entity resolution:**
```bash
hybrid-rag index ./fixtures/dependent_repo --repo-name main-app
```
*The resolver will query FalkorDB to find the real implementation of imported modules or inherited classes in existing namespaces (e.g. `core-lib`) and rewrite stubs to link across repos.*

---

## 🏛️ Graph Communities: Compiling Architecture summaries (Global Search)

Once indexed, compile vĩ mô architecture community summaries (Microsoft GraphRAG Option A) using the `hybrid-rag community-build` command:

```bash
hybrid-rag community-build [OPTIONS]
```

### Key Options
*   `--resolution <FLOAT>`: Modularity clustering resolution for Louvain (default: `1.0`). Higher values generate more smaller, finer-grained communities.
*   `--llm-model <MODEL>`: Local LLM used to compile structural summaries (default: `qwen2.5-coder:14b`).
*   `--graph-name <NAME>`: Targets a specific FalkorDB graph (default: `codebase`).

### Example
```bash
hybrid-rag community-build --resolution 1.0 --llm-model qwen2.5-coder:14b
```
*This splits the knowledge graph into communities, calls Qwen 2.5 Coder 14B to summarize each community locally on the M4 GPU, and persists these reports back to FalkorDB as `Community` nodes linked to codebase member nodes.*

---

## 🔍 Retrieval: Querying the Codebase

Use the `hybrid-rag query` command to ask natural language questions about the indexed codebase.

```bash
hybrid-rag query "<YOUR_QUESTION>" [OPTIONS]
```

### Key Options
*   `--repo-name <NAME>`: Scope the search. Only returns files and structural paths that belong to this repository namespace.
*   `--max-tokens <INT>`: Strict token budget for prompt context assembly.
*   `--max-chars <INT>`: Strict character budget for prompt context assembly.
*   `--top-k <INT>`: Number of candidates to retrieve initially from databases (default: `20`).

### Examples

**Global Cross-Repository Query:**
```bash
hybrid-rag query "Explain how SuperCalculator inherits Calculator"
```

**Scoped Repository Query (locks search to `core-lib`):**
```bash
hybrid-rag query "Explain how SuperCalculator inherits Calculator" --repo-name core-lib
```

**Token-Budget Context Assembly Query (limits prompt context to 1000 tokens):**
```bash
hybrid-rag query "Detail sparse retrieval logic" --max-tokens 1000
```

---

## 🌐 Web Visualization & Interactive API

Hybrid-RAG includes a state-of-the-art interactive REST API and a beautiful, high-fidelity developer-themed web explorer UI.

### Start the Server
```bash
hybrid-rag serve [OPTIONS]
```
*Starts the FastAPI server on `http://localhost:8000` (FastAPI Swagger docs are available at `http://localhost:8000/docs`).*

### Explore the Visualizer
Open your browser and navigate to:
```
http://localhost:8000/
```
The newly redesigned premium interface enables:
1.  **Dynamic Repository Scoping:** Select an active repository namespace (e.g. `core-lib`, `main-app`, or global `All Repositories`) from the dropdown. Both searches and RAG queries are instantly scoped to the selected repo.
2.  **Visual Hybrid Toggling (3D vs 2D):** Explore the codebase at macro level using **3D Force-Graph**, or seamlessly switch to a clean **2D Hierarchical/Cose Graph** (powered by Cytoscape.js) to trace call chains and Louvain community boundaries without hairball clutter.
3.  **Category Filtering Tabs:** Lock search inputs specifically to `Class`, `Function`, `Module`, `Variable`, or `Community` nodes via click tabs in the left sidebar.
4.  **Collapsible Floating Node Inspector:** Select a node in the graph to view its detailed FQN metadata, repository, and file paths. Perform quick actions directly:
    *   *Focus Node:* Animate camera/zoom to focus on the selected node.
    *   *Expand Neighbors:* Fetch and render the node's local relationships.
    *   *Ask AI:* Pre-populate and submit an inquiry about the node directly in the RAG Chat pane.
5.  **State-of-the-Art RAG Chat UX:** Submit structural/semantic RAG queries and get SSE streaming responses with latency counters and citation cards showing exact repository indicators.

---

## 📊 Evaluation & Diagnostics

Run milestones, latency benchmarks, and RAGAS evaluations.

### 1. Check Service Status
```bash
hybrid-rag status
```
*Verifies connection health and prints database node/point counts.*

### 2. Latency Benchmarks
```bash
hybrid-rag bench --corpus
```
*Measures p50/p95/p99 retrieval latency per query type.*

### 3. RAGAS Quality Evaluation
```bash
hybrid-rag ragas
```
*Computes RAGAS Faithfulness, Answer Relevancy, and Context Precision on the evaluation corpus.*

---

## 🐳 Containerized Stack Deployment (Private Machine)

For private servers, we package the RAG API and Visualizer along with the database engines using a single Docker Compose bundle.

### 1. Build and Start the Entire Stack
Copy the codebase to the private machine, navigate to the folder, and run:
```bash
docker compose up --build -d
```
This single command:
1. Compiles AST `tree-sitter` native bindings and builds the `hybrid-rag-api` image.
2. Starts FalkorDB, Qdrant, and the Hybrid-RAG API.
3. Automatically sets up connection paths.

### 2. Connect to Private Ollama
By default, the container routes to Ollama running natively on the physical host machine via:
`OLLAMA_BASE_URL=http://host.docker.internal:11434`

If you are running Ollama on a different server or IP address, simply edit `docker-compose.yml` to update the variable:
```yaml
environment:
  - OLLAMA_BASE_URL=http://<OLLAMA_SERVER_IP>:11434
```

### 3. Verification & Access
Once up and healthy, the services are accessible:
* **Interactive UI & Visualizer:** `http://localhost:8000/`
* **Swagger API Documentation:** `http://localhost:8000/docs`
* **MCP SSE Server:** `http://localhost:8001/sse`
* **Qdrant Dashboard:** `http://localhost:6333/dashboard`

### 4. Ingest / Index Repositories in Containerized Mode
Since the target codebase folders reside on your host filesystem (and are not mounted inside the Docker containers), you must run the indexing commands from your **host machine** (ensure your local virtual environment is active). The indexer will read the local files and write directly to the containerized database ports (`6379` and `6333` mapped on `localhost`):

```bash
# Ensure local virtual environment is active
source .venv/bin/activate

# 1. Index the Hybrid-RAG repo itself
hybrid-rag index . --repo-name hybrid-rag

# 2. Index all test fixtures (in correct order of dependencies for resolution)
hybrid-rag index ./fixtures/small_repo --repo-name small-app
hybrid-rag index ./fixtures/dependent_repo --repo-name main-app
hybrid-rag index ./fixtures/llama_index_core --repo-name llama-core

# 3. Build architectural communities for all indexed codebases
hybrid-rag community-build
```
Once indexed, the containerized REST API (`http://localhost:8000/`) and the containerized MCP server (`http://localhost:8001/sse`) will instantly have access to these codebases.
