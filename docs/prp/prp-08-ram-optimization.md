# Product Requirement Prompt (PRP) — Industrial-Grade RAM Optimization

## 🎯 Role & Objective
You are a principal backend and database engineer. Your task is to implement industrial-grade RAM and performance optimizations for the `hybrid-RAG` ingestion and query pipelines. The pipeline must scale to large codebases with constant memory footprints, and the API must use high-performance Redis caching, LLM concurrency controls, and active garbage collection to minimize CPU/VRAM load.

---

## 🏛️ Tech Stack & Service Dependencies
*   **Caching Store:** Redis (accessed via async `redis.asyncio` client).
*   **Inference Server:** Ollama (local only).
*   **Ingestion Pipeline:** Tree-sitter, FalkorDB, Qdrant.

---

## 🛠️ Functional Requirements

### 1. Streaming Batch Ingestion
*   **Eliminate Memory Leaks**: Avoid reading entire repository files into memory at startup (remove unused global allocations).
*   **Constant Ingestion RAM**: Process file chunking, embedding, and database uploads in sequential batches of 128 chunks.
*   **Immediate DB Upserts**: Do not accumulate embeddings in a global array. Call `vector_store.upsert(batch)` instantly, clear the batch payloads from memory, and run `gc.collect()` after each indexing session.

### 2. Redis Query Caching
*   **Unified Cache Manager**: Connect asynchronously to Redis (`REDIS_HOST`, `REDIS_PORT`) with a connection timeout (2.0s).
*   **Fail-Open Fallback**: If the Redis server is offline, down, or throws an Out-Of-Memory error, the caching layer must catch all exceptions, log warnings, and fall back to processing queries directly through RAG + LLM without crashing the API.
*   **Sync Q&A Caching**: Cache exact `/query` JSON outputs with a configurable TTL (default 1 hour).
*   **SSE Stream Caching**: Accumulate SSE streaming events (`data: {...}`) and cache the full stream list. When cached, replay the stream instantly to the client without calling the LLM.

### 3. Concurrency Guard (VRAM Semaphore)
*   **Single-Threaded Inference Lock**: Initialize a global `asyncio.Semaphore(1)` on API startup.
*   **Request Queuing**: Wrap all LLM generation calls (`_llm_generate` and `_llm_stream` token generation) within the semaphore lock. When multiple parallel requests arrive, queue them to prevent concurrent GPU VRAM allocations.

### 4. Dynamic Context Safety Cap
*   **VRAM Overflow Prevention**: Set a configurable maximum character limit (default: 12,000 characters, ~3,000 tokens) for retrieved context.
*   **Prompt Truncation**: Truncate context code blocks cleanly before assembling the final prompt to ensure prompt pre-fills never exceed local GPU memory capacities.

---

## 📈 Non-Functional Requirements & Performance
*   **O(1) Memory Footprint**: Python process RAM must remain constant during indexing and generation, regardless of repository size.
*   **Cache Hit Latency**: Cached query responses must resolve in less than 50ms (avoiding the 100+ second local LLM generation delay).
*   **Active GC**: Invoke `gc.collect()` at the end of heavy database writing and LLM streaming scopes.
