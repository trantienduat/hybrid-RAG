# Knowledge Graph Schema

Version: 0.2 — synchronized with the current Python parser and graph writers.

**Rule:** If a query in evaluation.md cannot be answered with this schema, schema must be updated first.

---

## Node Types

### `:Module`
Represents a Python file (`.py`).

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `id` | string | ✓ | Dotted module FQN, such as `hybrid_rag.api.main` |
| `name` | string | ✓ | Module name (e.g., `retriever_query_engine`) |
| `file_path` | string | ✓ | Repo-relative path (e.g., `llama_index/core/query_engine/retriever_query_engine.py`) |
| `language` | string | ✓ | Currently `python`; Java extraction is not implemented |
| `type` | string | ✓ | `source` \| `test` \| `init` |
| `line_count` | int | — | Total lines |

### `:Class`
Represents a class definition.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `id` | string | ✓ | Dotted FQN, such as `hybrid_rag.api.schemas.QueryRequest` |
| `name` | string | ✓ | Class name (e.g., `BaseRetriever`) |
| `file_path` | string | ✓ | Source file |
| `line_start` | int | ✓ | Start line in file |
| `line_end` | int | ✓ | End line in file |
| `docstring` | string | — | First docstring if present |
| `is_abstract` | bool | — | Has `ABC` or `abstractmethod` |

### `:Function`
Represents a function or method definition.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `id` | string | ✓ | Dotted function or method FQN |
| `name` | string | ✓ | Function name |
| `file_path` | string | ✓ | Source file |
| `class_name` | string | — | Parent class name (null for module-level fns) |
| `line_start` | int | ✓ | Start line |
| `line_end` | int | ✓ | End line |
| `signature` | string | ✓ | Full signature string (e.g., `def retrieve(self, query: str) -> List[NodeWithScore]`) |
| `docstring` | string | — | First docstring |
| `is_async` | bool | ✓ | `async def` or not |
| `is_abstract` | bool | — | Decorated with `@abstractmethod` |
| `is_property` | bool | — | Decorated with `@property` |

### Additional node types

| Label | Purpose |
|-------|---------|
| `Directory` | Repository directory hierarchy derived from parsed file paths |
| `Document` | Markdown files when `NON_CODE_INGESTION=true` |
| `Configuration` | YAML and Dockerfile content when non-code ingestion is enabled |
| `Community` | Directory-based architectural summary |
| `RepositoryMetadata` | Last indexed Git commit and synchronization timestamp |

`Variable` remains a reserved relationship target label for compatibility with
LLM-extracted `USES` edges. The current AST parser does not emit Variable nodes.

---

## Edge Types

### `[:IMPORTS]`
Module imports another module.

```
(:Module)-[:IMPORTS {alias: string?, is_from: bool}]->(:Module)
```

| Property | Description |
|----------|-------------|
| `alias` | Import alias if any (`import x as y`) |
| `is_from` | True if `from x import y` style |

### `[:DEFINES]`
Module/Class defines a Class/Function.

```
(:Module)-[:DEFINES]->(:Class)
(:Module)-[:DEFINES]->(:Function)
(:Class)-[:DEFINES]->(:Function)
(:Directory)-[:DEFINES]->(:Directory)
(:Directory)-[:DEFINES]->(:Module)
```

No properties.

### `[:INHERITS]`
Class inherits from another class.

```
(:Class)-[:INHERITS {order: int}]->(:Class)
```

| Property | Description |
|----------|-------------|
| `order` | MRO position (1 = primary base) |

### `[:CALLS]`
Function calls another function.

```
(:Function)-[:CALLS {line: int, callee_expr: str}]->(:Function)
```

| Property | Description |
|----------|-------------|
| `line` | Line number of the call site |
| `callee_expr` | Full call expression text (e.g. `self.retrieve`) |

### `[:USES]`
Function references a user-defined class or optional LLM-extracted target.

```
(:Function)-[:USES]->(:Class)
```

No properties.

### `[:DEFINED_IN]`
Class or Function belongs to a Module (redundant but speeds up traversal).

```
(:Class)-[:DEFINED_IN]->(:Module)
(:Function)-[:DEFINED_IN]->(:Module)
```

No properties.

---

## Schema Diagram (ASCII)

```
Module ──[:IMPORTS]──────────────────────► Module
  │                                           │
  └──[:DEFINES]──► Class ──[:DEFINES]──► Function ──[:CALLS]──► Function
                     │         │                │
                     │
                     └──[:INHERITS]──► Class
                     │
                     └──[:DEFINED_IN]──► Module
```

---

## Entity Resolution Rules

Deduplication key per node type:

| Node | Key | Rule |
|------|-----|------|
| Module | dotted module FQN | Exact match |
| Class | dotted class FQN | Exact match |
| Function | dotted function or method FQN | Exact match |
| Directory | repository-relative directory path | Exact match |

**Cross-file class resolution:** When `[:INHERITS]` target is from external import, resolve via `[:IMPORTS]` chain to find actual `file_path`. If unresolvable (stdlib/third-party), create stub node with `type: external`.

---

## Scope Boundaries

**In scope for extraction:**
- All `.py` files in the target repository
- Markdown, YAML, and Dockerfiles when non-code ingestion is enabled
- Third-party classes appear as stub nodes `{type: "external"}` — no properties beyond name
- Java language requests and Java-only repositories are rejected before indexing

**Out of scope:**
- Runtime dynamic attributes (`setattr`, `__dict__`)
- Lambda functions (anonymous — no stable ID)
- Decorator internals (decorators recorded as metadata on Function node, not as separate nodes)

---

## Version History

| Version | Change |
|---------|--------|
| 0.1 | Initial schema derived from 20 test queries |
| 0.2 | Document FQN IDs and current directory, document, community, and metadata nodes |
