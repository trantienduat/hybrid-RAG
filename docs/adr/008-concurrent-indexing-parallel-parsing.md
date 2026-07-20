# ADR-008: Concurrent Repository Indexing and Parallel AST Parsing

## Status
Accepted

## Context

The original indexing pipeline used a single `asyncio.Lock` to serialize all indexing tasks — only one repository could be indexed at any time. With 12 repositories in the system, this meant total queue-flush time was the sum of all individual indexing times (wall-clock serial execution). For large codebases like `transformers` (3,500+ Python files) or `langchain-core`, a single full rebuild could take several minutes, blocking the entire queue.

Additionally, within a single repository, source files were parsed **sequentially** in a `for` loop. Tree-sitter is a C extension with its own native parser; it releases Python's GIL during parsing, which means sequential Python-level iteration was leaving CPU cores idle.

## Decision

Apply two complementary concurrency improvements:

### 1. Concurrent Repository Indexing (Semaphore)

Replace `asyncio.Lock()` with `asyncio.Semaphore(3)` as the indexing concurrency guard:

```python
# Before
app.state.indexing_lock = asyncio.Lock()      # 1 concurrent task

# After
app.state.indexing_lock = asyncio.Semaphore(3) # up to 3 concurrent tasks
```

- Up to **3 repositories** can run their full indexing pipelines concurrently.
- The Semaphore value `3` was chosen empirically: it balances FalkorDB write throughput (Redis single-threaded command execution) against CPU/memory pressure on the indexing host (Apple M2 Max, 12-core).
- Tasks beyond the concurrency limit queue normally and are dispatched as slots free up.

### 2. Parallel AST Parsing (ThreadPoolExecutor)

Inside `parse_repo`, replace the sequential `for fpath in files: parse_file(fpath)` loop with a `ThreadPoolExecutor`:

```python
max_workers = min(32, (os.cpu_count() or 4) * 2)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = executor.map(parse_single_file, files_to_parse)
```

- `parse_file` calls tree-sitter's C extension, which **releases the GIL** during native parsing. True parallel execution is achieved across physical CPU cores.
- `max_workers = 2 × CPU_count` is capped at 32 to avoid thread explosion on machines with large core counts.
- Directory node collection (single-pass `rglob`) and post-parse `DEFINES` edge generation are performed on the main thread before and after the executor, respectively, keeping shared state mutation thread-safe.

## Alternatives Considered

| Alternative | Reason Rejected |
|-------------|----------------|
| Keep `asyncio.Lock` (serial) | Unacceptably slow for 12-repo fleet; total rebuild time scales linearly |
| Semaphore(∞) / no limit | Risk of FalkorDB write conflicts and OOM from concurrent Ollama embedding calls |
| `ProcessPoolExecutor` for parsing | Higher IPC overhead (pickling tree-sitter objects); threading sufficient since GIL is released |
| Async file I/O (`aiofiles`) | Tree-sitter parse is CPU-bound, not I/O-bound; async I/O would not help the bottleneck |

## Rationale

- **Semaphore(3) as a soft limit** is preferable to a hard lock because it allows the system to exploit available parallelism while bounding resource usage. The value `3` can be tuned via config without code changes.
- **Threading over multiprocessing** is the correct choice for tree-sitter because the GIL is not the bottleneck — the C extension releases it. Threads share the same process memory, avoiding serialization overhead.
- **Single-pass directory collection** (integrated into the file-collection loop) eliminates the earlier double-`rglob` pass (once to parse files, once to collect directories), halving filesystem I/O.

## Consequences

- Concurrent FalkorDB writes from 3 simultaneous pipelines increase the probability of transient write conflicts. FalkorDB (Redis-based) serializes commands internally, so correctness is maintained, but latency per write may increase under high concurrency.
- Qdrant vector upserts from concurrent repos are independent per collection; no write conflicts expected.
- Thread safety: `parse_file` is stateless (reads only, no shared mutable state). The `ThreadPoolExecutor` is safe as long as no global mutable state is modified inside `parse_file` — confirmed by inspection.
- If Ollama is unavailable or slow, the embedding phase (run after parse) becomes the new bottleneck; the parallel parse optimization does not help the embedding phase directly.
