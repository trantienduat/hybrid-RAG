# ADR-007: Multi-Repository Master Graph with Hierarchical Drill-Down Navigation

## Status
Accepted

## Context

The original graph visualization portal was scoped to a **single repository at a time**: the user selected a repo from a dropdown, and the graph rendered that repo's AST nodes (Module, Class, Function) and their relationships. This model had several limitations:

1. **No global view**: Engineers working across microservices, monorepos, or shared libraries had no way to see cross-repository relationships or compare codebases side-by-side.
2. **Flat navigation**: After entering a repository, the user was presented with potentially thousands of raw AST nodes with no meaningful entry point — no architectural grouping, no hierarchy.
3. **Navigation loss**: Drill-downs were stateless; the user had no way to return to a previous level without refreshing.
4. **Python-only repos invisible**: Repos containing only `.py` files had no `Directory` nodes (they were only created during non-code file parsing), making them appear as a single dot in the global view.

## Decision

Implement a **multi-repository master graph** exposed via `GET /graph/master` and a **4-level hierarchical drill-down navigation** model in the web UI.

### Navigation Levels
```
Level 0: Master Graph     — all RepositoryMetadata + top-level Directory nodes (all repos)
Level 1: Community Graph  — Louvain community clusters within a selected repository
Level 2: Node Neighbors   — direct graph neighbors of a selected Module/Class/Function
Level 3+: Deep Traversal  — recursive neighbor expansion from any code node
```

### API Changes
- `GET /graph/master`: Returns all `RepositoryMetadata` nodes and their top-level `Directory` nodes, filtered at the Cypher level to exclude noise directories (`node_modules`, `.venv`, `.agents`, `__pycache__`, `.git`, etc.).
- `GET /graph/repositories/{repo}/community`: Returns `Community` nodes for that repo (Level 1).
- `GET /graph/neighbors/{node_id}`: Returns immediate neighbors of any node (Levels 2+).

### Schema Extensions
- **`Directory` nodes**: Generated from *all* parsed source file paths inside `parse_repo`, ensuring every Python-only repo has directory structure nodes.
- **`WEAK_LINK` edges**: Connect each `Directory` to its parent `Directory` or `RepositoryMetadata`.
- **`DEFINES` edges (Directory → Module)**: Created post-parse from each Module's `file_path`, enabling directory expansion in the UI to reveal contained modules.

### Frontend
- **`navStack` breadcrumb bar**: Persistent breadcrumb records the full drill-down path; any crumb is clickable to jump back.
- **`isMasterView` flag**: Controls whether node clicks expand inline (master view) or navigate to community/neighbor view.
- **Cypher-level noise filter**: `STARTS WITH` and `CONTAINS` predicates in the `get_master_graph` query prevent stale noise nodes from appearing.

## Alternatives Considered

| Alternative | Reason Rejected |
|-------------|----------------|
| Per-repo portal only (status quo) | No cross-repo view; scaling to 12+ repos is impractical |
| URL-based SPA routing | Added significant framework complexity; single-file HTML was sufficient |
| OS `os.walk` for Directory node generation | Would create nodes for excluded paths unless carefully filtered; derivation from parsed file paths is simpler and always consistent |

## Rationale

- **Cypher-level noise filtering** reduces network transfer and avoids exposing stale nodes through the API, rather than filtering post-query in Python.
- **Deriving Directory nodes from parsed file paths** ensures only directories containing *indexed* source files appear — no phantom directories from excluded paths.
- **Breadcrumb navigation** keeps all navigation state explicit in a `navStack` array rather than relying on browser history or modal states.

## Consequences

- Requires a full `rebuild=True` re-index of all repositories after deployment to populate `Directory` and `DEFINES` edges for Python-only repos.
- `GET /graph/master` performs runtime Cypher filtering; for installations with 100+ repos, pagination or a cached materialized view may be needed.
- New repositories automatically appear in the master graph after their first indexing cycle — no manual registration required.
- `Directory` nodes are now first-class citizens in the FalkorDB schema alongside `Module`, `Class`, `Function`, `Community`, and `RepositoryMetadata`.
