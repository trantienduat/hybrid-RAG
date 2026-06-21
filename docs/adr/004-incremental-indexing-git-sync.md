# ADR-004: File-level Incremental Indexing & Git Sync

## Status
Accepted

## Context
Full re-indexing of a codebase on every update is computationally expensive and slow (takes several minutes for large repositories), which degrades the local developer experience and increases API/Ollama costs.
We need an incremental indexing strategy that:
1. Detects added, modified, renamed, and deleted files compared to the last indexed state.
2. Performs scoped, precise cleanup of existing data at the file level in both the Graph Store (FalkorDB) and Vector Store (Qdrant).
3. Keeps track of the indexing state in a centralized way to support multiple developers and CI/CD pipelines without local cache files.
4. Integrates seamlessly with Git workflows (hooks) and CI/CD parameters.

## Decision
Implement a **Git-driven, Database-centric Incremental Indexing Engine**:
1. **Change Detection**: Use `git diff --name-status <commit>` combined with `git status --porcelain` (to catch untracked files) to classify changes into `modified` and `deleted` sets.
2. **State Storage**: Store the last successfully indexed commit hash (`last_indexed_commit`) directly in FalkorDB inside a `RepositoryMetadata` node.
3. **Scoped Cleanup**:
   - In FalkorDB: Use Cypher `MATCH (n) WHERE n.file_path = $file_path AND n.repository = $repo DETACH DELETE n` to purge nodes associated with modified/deleted files.
   - In Qdrant: Use payload filters (`file_path` and `repository`) to delete corresponding vector points.
4. **CLI Flags**: Add `--incremental`, `--rebuild`, and `--from-commit <hash>` flags to the CLI.
5. **Git Hooks**: Provide a `hybrid-rag install-hooks` command to write `post-merge` and `post-checkout` shell scripts that trigger incremental indexing.

## Rationale
- **Database as Source of Truth**: Storing the last-indexed commit in FalkorDB ensures that anyone checking out the codebase (or a CI runner) shares the same index state, avoiding sync issues with local cache files.
- **Precise Cleanup**: Cypher's `DETACH DELETE` and Qdrant's payload selector filter delete are fast and guarantee that old nodes/vectors are completely cleared before fresh indexing, avoiding orphans and duplicates.
- **Git Hook Automation**: Running incremental indexing automatically on checkout/pull ensures that the RAG model is always up to date with the local code branch.

## Consequences
- Requires a working Git environment in the repository folder.
- If the repository has untracked changes, they are indexed as "modified".
- If the database is wiped or rebuilt, the `RepositoryMetadata` node is lost, triggering a fallback to a clean Full Index.
- CI/CD pipelines can override the auto-detected last commit using the `--from-commit` option (e.g. from GitHub Actions change events).
