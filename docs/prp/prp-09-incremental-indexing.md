# Product Requirement Prompt (PRP) — Incremental Indexing & Git Sync

## 🎯 Role & Objective
You are a senior systems engineer and DevOps specialist. Your task is to implement an enterprise-grade **Incremental Indexing Engine** and **Git Synchronization Hook system** for the `hybrid-RAG` platform. The engine must minimize API usage, LLM costs, and parsing latency by targeting only added, modified, renamed, and deleted source files, while maintaining a single, consistent graph and vector index state across all developers.

---

## 🏛️ System Design & Database Schema
*   **Graph Store (FalkorDB):** 
    *   Metadata is stored in a dedicated node: `(r:RepositoryMetadata {id: $repo})`.
    *   Properties: `last_indexed_commit` (string), `updated_at` (timestamp).
    *   File-level nodes have a `file_path` property (string) and a `repository` property (string).
*   **Vector Store (Qdrant):**
    *   Each point payload includes `file_path` (string) and `repository` (string) fields.

---

## 🛠️ Functional Requirements

### 1. Git-Based Change Detection
*   Using subprocesses, identify the current Git `HEAD` commit.
*   Identify the previous indexed commit from FalkorDB.
*   Run `git diff --name-status <last_commit>` to extract modified/added (`M`, `A`, `R`) and deleted (`D`) files.
*   Run `git status --porcelain` to identify untracked changes (`??`).
*   Group these into two sets: `modified_files` and `deleted_files`.
*   Filter out files that do not match target file extensions (e.g. non-python files) or are excluded by pattern.

### 2. Precise Store Cleanup (Healing Phase)
*   Before indexing any new data for modified or deleted files, purge existing nodes and vectors associated with those specific files.
*   **Graph store purge**: Use Cypher `DETACH DELETE` matching the `file_path` and `repository` variables.
*   **Vector store purge**: Use Qdrant payload filters matching `file_path` and `repository`.

### 3. Scoped AST Parsing & Resolution
*   Parse AST structures only for files in the `modified_files` set.
*   Resolve entity stubs locally and globally (cross-referencing with the rest of the graph already in FalkorDB) to update cross-file references.
*   Generate embeddings only for chunks coming from modified files and upsert them.

### 4. CLI Controls & Hook Automation
*   Extend the CLI `index` command to accept:
    *   `--incremental/--no-incremental` (default `True`).
    *   `--rebuild` (ignores metadata, forces full re-index).
    *   `--from-commit` (overrides stored commit for manual range comparison).
*   Implement `install-hooks` command to write shell scripts to `.git/hooks/post-merge` and `.git/hooks/post-checkout` that execute `hybrid-rag index . --incremental` automatically.

---

## 📈 Non-Functional Requirements & Safety
*   **Safe Fallback**: If Git commands fail, if the folder is not a Git repository, or if `last_indexed_commit` is missing, fall back to a full rebuild.
*   **No Orphans**: Ensure that deleting file-level nodes does not leave disconnected dangling relationships. Use Cypher `DETACH DELETE` to automatically prune associated edges.
*   **Consistency**: Ensure that a commit state is only saved to the database *after* both the FalkorDB and Qdrant ingestion runs complete successfully.
