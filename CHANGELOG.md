# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `embed_texts` method to `BaseEmbedder` interface to support batch text embedding with a default sequential fallback for backward compatibility.
- Implemented concurrent `embed_texts` in `OllamaEmbedder` utilizing a `ThreadPoolExecutor` with a configurable concurrency limit (defaults to `EMBED_CONCURRENCY=8`) to prevent hardware resource exhaustion.
- Added comprehensive unit tests in `tests/unit/test_ollama_embedder.py` covering embedding single queries, concurrent batch execution, sequential fallbacks, stubs filtering, and error handling.

### Changed
- Refactored the embedding phase in `cli.py` to pre-collect all AST chunks prior to embedding, enabling batch ingestion with a fixed size of 128.
- Improved the CLI user experience by updating the Rich progress bar smoothly per batch of 128, eliminating terminal spinner freeze during large repository ingestion.

## [0.1.0] - 2026-05-24

### Added
- Created the core AST extraction engine powered by tree-sitter supporting Python and Java.
- Implemented local Graph Database integration using FalkorDB for structural relational data (Class, Function, Variable, Module nodes, and INHERITS, IMPORTS, CALLS, DEFINES edges).
- Implemented Vector Database integration using Qdrant for semantic chunk retrieval.
- Implemented hybrid RAG search fusing structural and semantic signals using Reciprocal Rank Fusion (RRF).
- Added robust error handling and failsafes in `OllamaEmbedder` including 8k character truncation, 3x exponential backoff retries, a 500-character deep fallback, and a permanent `[0.0] * 768` zero-vector fallback.
- Added evaluation suite featuring RepoQA benchmark expansion to 100 cases, graph fidelity assertion metrics, and automated thesis chart generators.
