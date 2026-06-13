# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-13

### Added
- Created a persistent **Conversation History Manager** in the Web UI sidebar using browser `localStorage`, allowing users to save, switch between, and delete chat threads.
- Added a frontend **LLM Generation Stop Button** using `AbortController` to cancel active streaming HTTP requests and free backend resources.
- Implemented **Dynamic Resizable Layout Panels** with vertical divider bars (`.resizer`) that support responsive drag-to-resize operations and trigger automatic Cytoscape & 3D Force-graph resizing.
- Implemented a **Collapsible Reasoning (Thinking Box)** to extract model thoughts wrapped inside `<think>...</think>` tags and present them in a glassmorphic details panel.
- Added an execution **Trace & Metadata Dashboard** rendering badges for query classification (HYBRID/VECTOR/etc.), retrieved source count, and fusion status with generation latency.
- Integrated `marked.js` library via CDN for rich Markdown rendering of LLM responses (headings, tables, bullet lists, code blocks).
- Added recent search history tags and node type suggestion chips to the sidebar.
- Added a Product Requirement Prompt (`docs/prp/prp-07-ui-ux-enhancements.md`) specifying frontend improvements and master replication prompts.

### Changed
- Increased server-side HTTPX timeouts (`_llm_stream` and `_llm_generate` in `src/hybrid_rag/api/main.py`) from `120s` to `300s` to support long codebase prompt pre-fills under local LLMs.
- Updated system prompts in `src/hybrid_rag/api/main.py` with instructions for Gemma 4 to output step-by-step reasoning inside `<think>` tags and handle sparse codebase contexts conceptually.

### Fixed
- Fixed a critical chat history ID collision bug in `index.html` where loading a saved chat thread failed to update the global `msgIdCounter`, causing subsequent query responses to overwrite previous assistant bubbles.

## [0.2.0] - 2026-05-28

### Added
- Added `embed_texts` method to `BaseEmbedder` interface to support batch text embedding with a default sequential fallback for backward compatibility.
- Implemented concurrent `embed_texts` in `OllamaEmbedder` utilizing a `ThreadPoolExecutor` with a configurable concurrency limit (defaults to `EMBED_CONCURRENCY=8`) to prevent hardware resource exhaustion.
- Added comprehensive unit tests in `tests/unit/test_ollama_embedder.py` covering embedding single queries, concurrent batch execution, sequential fallbacks, stubs filtering, and error handling.
- Added fully FQN-aware node ID generation in `parser.py` using dot-notation (e.g. `module.class.func`) to completely eliminate class and function name collisions.
- Added helper `rel_path_to_fqn` to parse package structure cleanly from file paths in Python and Java.

### Changed
- Refactored the embedding phase in `cli.py` to pre-collect all AST chunks prior to embedding, enabling batch ingestion with a fixed size of 128.
- Improved the CLI user experience by updating the Rich progress bar smoothly per batch of 128, eliminating terminal spinner freeze during large repository ingestion.
- Refactored `entity_resolver.py` to resolve class and module stubs using FQN dot-notation mappings.
- Refactored `runner.py` RepoQA name parsing and matching, `main.py` neighbor explorer, and static frontend explorers to extract FQN simple names flawlessly.
- Updated all unit and integration test assertions to align with FQN module, class, and function node ID structures.

### Performance & Quantitative Improvements
- **Indexing Speedup (Action Item 1):** Reduced the embedding and indexing time for a large codebase like `huggingface/transformers` (49k chunks) from **45.6 minutes to ~6.5 minutes** (an **85.7% decrease in indexing time / 7x speedup**) using 8 concurrent workers. Small repository indexing completed in just **1.4 seconds**.
- **Edge Collision Rate (Action Item 2):** Resolved class/method name collisions in large repositories (previously causing up to 12 edge mismatches on cross-file class resolutions in `llama_index`) down to **absolute 0** (a **100% collision rate reduction**), ensuring 100% Graph Fidelity.

## [0.1.0] - 2026-05-24

### Added
- Created the core AST extraction engine powered by tree-sitter supporting Python and Java.
- Implemented local Graph Database integration using FalkorDB for structural relational data (Class, Function, Variable, Module nodes, and INHERITS, IMPORTS, CALLS, DEFINES edges).
- Implemented Vector Database integration using Qdrant for semantic chunk retrieval.
- Implemented hybrid RAG search fusing structural and semantic signals using Reciprocal Rank Fusion (RRF).
- Added robust error handling and failsafes in `OllamaEmbedder` including 8k character truncation, 3x exponential backoff retries, a 500-character deep fallback, and a permanent `[0.0] * 768` zero-vector fallback.
- Added evaluation suite featuring RepoQA benchmark expansion to 100 cases, graph fidelity assertion metrics, and automated thesis chart generators.
