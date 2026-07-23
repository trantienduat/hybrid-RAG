# Hybrid-RAG

Hybrid-RAG indexes Python codebases into a FalkorDB knowledge graph and a
Qdrant vector collection, then combines structural and semantic retrieval for
CLI, REST, web UI, and MCP clients.

The default Ollama path runs locally. Optional Gemini providers send source
text, queries, embeddings, or assembled context to Google's API.

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

## Current limitations

- Python AST extraction is implemented. Java is accepted by configuration but
  its AST-to-graph extraction is not implemented yet.
- The CLI `query` command prints retrieved context; answer generation is
  available through `/query`, `/query/stream`, and the web UI.
- Live integration tests require FalkorDB, Qdrant, and Ollama.
