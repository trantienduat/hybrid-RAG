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

Hybrid-RAG includes a beautiful REST API and an interactive D3.js force-directed graph UI.

### Start the Server
```bash
hybrid-rag serve [OPTIONS]
```
*Starts the FastAPI server on `http://localhost:8000` (FastAPI docs available at `http://localhost:8000/docs`).*

### Explore the Visualizer
Open your browser and navigate to:
```
http://localhost:8000/
```
The interface allows you to:
1.  **Visualize:** Zoom, pan, and drag nodes in the live Knowledge Graph.
2.  **Inspect:** Click on any node (Module, Class, Function) to inspect its Fully Qualified Name (FQN), properties, and structural links.
3.  **Chat:** Run queries in the right panel and see live vector citations highlighted in green on the graph.

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
* **Qdrant Dashboard:** `http://localhost:6333/dashboard`
