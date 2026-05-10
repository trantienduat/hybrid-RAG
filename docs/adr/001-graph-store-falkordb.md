# ADR-001: Graph Store — FalkorDB

## Status
Accepted

## Context
Need a graph database supporting:
- Cypher query language (familiar, expressive for graph traversal)
- Local deployment only (privacy requirement — no cloud)
- High performance for real-time retrieval (sub-second target)
- LlamaIndex native integration

Candidates evaluated:
- **Neo4j** — industry standard, excellent Cypher, but community edition limited; Docker image large (~1GB)
- **FalkorDB** — C-based sparse matrix engine, Redis-protocol compatible, Cypher support, ~200MB image
- **ArangoDB** — multi-model, AQL not Cypher, less LlamaIndex support
- **NetworkX** — in-memory Python only, no persistence, no Cypher

## Decision
Use **FalkorDB** via Docker.

## Rationale
- LlamaIndex has `llama-index-graph-stores-falkordb` — direct PropertyGraphIndex support
- Redis protocol means standard `redis` Python client works as fallback
- Significantly lighter than Neo4j on M4 Mac (important — shared RAM with Ollama)
- Cypher support covers all traversal patterns in evaluation.md queries
- Smoke test passed: PONG response confirmed on port 6379

## Consequences
- Must use FalkorDB-compatible Cypher subset (no APOC procedures)
- Less community resources than Neo4j — debugging harder
- Data persisted to Docker volume `falkordb_data`
- If FalkorDB proves insufficient → migration path to Neo4j is feasible (same Cypher)
