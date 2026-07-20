# ADR-009: Directory Nodes Derived from Parsed Code File Paths

## Status
Accepted

## Context

The FalkorDB graph schema originally had no `Directory` nodes for repositories that contained only code files (`.py`, `.java`). `Directory` nodes were exclusively created by the non-code ingestion pipeline — i.e., when a `.yaml`, `.md`, or `Dockerfile` was parsed, the code would walk up the file's directory hierarchy and emit `Directory` nodes and `WEAK_LINK` edges as a side effect.

This design had two consequences:

1. **Python-only repos were structurally invisible**: Repos such as `langchain-core`, `llama-core`, and `transformers` — which are exclusively Python — had no `Directory` nodes at all. In the master graph portal, they appeared as a single `RepositoryMetadata` node with no expandable structure.

2. **Module nodes were disconnected from Directory nodes**: The `_extract_python` extractor created `Module` nodes with a `file_path` property but emitted no edge connecting the Module to its parent directory. There was no graph path from `RepositoryMetadata → Directory → Module`; only `RepositoryMetadata → (implicit, via embedding)`.

## Decision

Add two post-parse steps inside `parse_repo` that derive structural graph data from the set of already-filtered source file paths:

### Step 1: Directory Node Generation (single-pass, before parsing)

During the file-collection loop (before the `ThreadPoolExecutor` parse phase), walk each source file's path upward to its root and emit `Directory` nodes + `WEAK_LINK` edges:

```python
def process_fpath_dirs(fpath: Path):
    current = Path(*rel_parts[:-1])
    while current and str(current) != ".":
        dir_id = f"{repo_name}::dir::{current}"
        dir_nodes.append(NodeData(label="Directory", id=dir_id, ...))
        parent = current.parent
        if str(parent) == ".":
            dir_edges.append(EdgeData(src_id=dir_id, rel="WEAK_LINK", dst_id=repo_name))
        else:
            dir_edges.append(EdgeData(src_id=dir_id, rel="WEAK_LINK", dst_id=f"{repo_name}::dir::{parent}"))
        current = parent
        if dir_str in seen_dirs: break  # memoize
```

The `seen_dirs` set ensures each directory is emitted only once regardless of how many files share it.

### Step 2: Directory → Module DEFINES Edges (post-parse)

After the `ThreadPoolExecutor` completes and all `Module` nodes are available in `combined.nodes`, iterate over them and emit `DEFINES` edges from their parent `Directory`:

```python
for node in combined.nodes:
    if node.label != "Module": continue
    parent_dir = str(Path(node.properties["file_path"]).parent)
    if parent_dir == ".":
        src_id = repo_name  # top-level: link to RepositoryMetadata
    else:
        src_id = f"{repo_name}::dir::{parent_dir}"
    combined.edges.append(EdgeData(src_id=src_id, rel="DEFINES", dst_id=node.id))
```

This makes the graph traversal `RepositoryMetadata → Directory → Module → Class/Function` fully navigable.

## Alternatives Considered

| Alternative | Reason Rejected |
|-------------|----------------|
| OS `os.walk` to enumerate directories | Would include excluded directories (`node_modules`, `.venv`) unless filtered separately; derivation from already-filtered file paths is simpler and always correct |
| Emit `Directory` nodes inside `parse_file` for `.py` files | Would require modifying `parse_file` signature and break the single-responsibility principle; `parse_repo` already has access to all file paths |
| Rely on non-code ingestion to create directories | Non-code ingestion is optional (`NON_CODE_INGESTION=false` by default); this would silently break Python-only repos |
| Generate `DEFINES` edges inside `_extract_python` | `_extract_python` does not have access to the `repo_name` prefix needed to construct the `Directory` node ID |

## Rationale

- **File-path derivation is always consistent** with what was actually indexed: a `Directory` node is only created if at least one source file under it was successfully parsed and included in the result. There are no phantom directories.
- **`seen_dirs` memoization** makes the directory walk O(depth × unique_dirs) rather than O(files × depth), keeping overhead minimal even for repos with thousands of files.
- **Post-parse `DEFINES` edges** are emitted as a separate step (not inside `parse_file`) to keep `parse_file` stateless and parallelism-safe — the `ThreadPoolExecutor` parse phase has no shared mutable state.
- **`WEAK_LINK` vs `DEFINES`**: `WEAK_LINK` captures the structural containment between directories (directory hierarchy). `DEFINES` captures the semantic containment of a Module within its parent directory — consistent with the existing schema where `Module DEFINES Class` and `Class DEFINES Function`.

## Consequences

- All repositories (including Python-only ones) now produce `Directory` nodes and are fully navigable in the master graph portal after a full `rebuild=True` re-index.
- The number of edges in FalkorDB increases significantly for large repositories (e.g., `transformers` gains ~3,500 `DEFINES` edges and ~200 `WEAK_LINK` edges). This is expected and manageable for FalkorDB's Redis-backed graph engine.
- Incremental indexing (`rebuild=False`) only processes changed files via `parse_file` directly, bypassing `parse_repo`. This means Directory nodes and DEFINES edges are only refreshed on full rebuild. For incremental sync, newly added files will have their Module nodes created but may lack a corresponding `DEFINES` edge until the next full rebuild — an acceptable trade-off for now.
- Any future language parser (TypeScript, C++) must also produce Module nodes with a `file_path` property for this mechanism to work automatically.
