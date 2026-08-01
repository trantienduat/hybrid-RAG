# Hybrid-RAG

Hybrid-RAG indexes Python codebases into a FalkorDB knowledge graph and a
Qdrant vector collection, then combines structural and semantic retrieval for
CLI, REST, web UI, and MCP clients.

All embedding and answer generation runs through a local Ollama service; the
runtime has no cloud-provider path.

## Quick start

```bash
make install
source .venv/bin/activate
ollama pull nomic-embed-text
ollama pull gemma4:12b
docker compose up -d --build
hybrid-rag index . --repo-name hybrid-rag
hybrid-rag query "How does incremental indexing work?" --repo-name hybrid-rag
```

The default stack exposes the web API on port 8000 and MCP over SSE on port
8001. Enable the optional observability stack with:

```bash
docker compose --profile observability up -d --build
```

See [operations](docs/usage.md), [architecture](docs/architecture.md), and
[MCP setup](docs/mcp.md) for details.

## API concurrency benchmark

Run the controlled before/after benchmark for blocking retrieval:

```bash
.venv/bin/python scripts/benchmark_api_concurrency.py \
  --requests 20 --delay-ms 50 --trials 7
```

The JSON output reports batch latency, throughput, completion p50/p95, and
worst event-loop stall. Results depend on the host CPU and scheduler; compare
runs made with the same parameters and environment.

Measure a repository's Python token footprint and an explicit hypothetical
hosted-input-cost scenario with:

```bash
.venv/bin/python scripts/benchmark_token_savings.py . \
  --extensions .py --rag-size 3000 --queries 1000 --json
```

The file/token counts are measured. RAG prompt size and hosted price are input
assumptions; the output does not claim actual Ollama cost or answer quality.

## Current limitations

- Python AST extraction is implemented. Java repositories and language options
  are rejected until Java extraction is implemented.
- The CLI `query` command prints retrieved context; answer generation is
  available through `/query`, `/query/stream`, and the web UI.
- Live integration tests require FalkorDB, Qdrant, and Ollama.
