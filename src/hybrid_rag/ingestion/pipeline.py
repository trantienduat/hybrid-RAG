"""
Unified indexing pipeline for Hybrid-RAG.

Allows running the parsing, extraction, resolution, and ingestion steps
programmatically with progress callbacks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from hybrid_rag.config import validate_local_ollama_url, validate_local_provider_configuration
from hybrid_rag.constants import INDEX_SCHEMA_VERSION
from hybrid_rag.ports import GraphStore, VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_BATCH_SIZE = 128


def _embedding_batch_size() -> int:
    """Read the embedding batch size, rejecting values that cannot make progress."""
    raw_value = os.environ.get("EMBED_BATCH_SIZE", str(_DEFAULT_EMBED_BATCH_SIZE))
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise ValueError("EMBED_BATCH_SIZE must be a positive integer") from exc
    if batch_size < 1:
        raise ValueError("EMBED_BATCH_SIZE must be a positive integer")
    return batch_size


def _validate_python_repository(
    repo: Path, languages: list[str], excludes: list[str] | None
) -> None:
    """Reject unsupported languages and Java-only projects before store access."""
    if languages != ["python"]:
        raise ValueError("Only Python indexing is currently supported; Java support is deferred")

    exclude_set = (
        set(excludes)
        if excludes is not None
        else {
            ".venv",
            "venv",
            "fixtures",
            "experiments",
            "dist",
            "build",
            ".git",
            "__pycache__",
            "node_modules",
            ".agents",
            ".pytest_cache",
            ".ruff_cache",
            ".roo",
            ".clinerules",
        }
    )

    def has_source(suffix: str) -> bool:
        return any(
            not any(part in exclude_set or part.startswith(".venv") for part in path.parts)
            for path in repo.rglob(f"*{suffix}")
        )

    if not has_source(".py") and has_source(".java"):
        raise ValueError("Java indexing is not implemented; this repository has no Python sources")


def _ensure_repository_node(result: Any, repo_name: str) -> None:
    """Ensure root graph relationships always have a concrete endpoint."""
    if any(node.label == "RepositoryMetadata" and node.id == repo_name for node in result.nodes):
        return
    from hybrid_rag.ingestion.parser import NodeData

    result.nodes.insert(
        0,
        NodeData(
            label="RepositoryMetadata",
            id=repo_name,
            properties={"name": repo_name, "repository": repo_name},
        ),
    )


def _metadata_is_compatible(metadata: dict[str, Any], embed_model: str) -> bool:
    return (
        metadata.get("index_schema_version") == INDEX_SCHEMA_VERSION
        and metadata.get("embedding_model") == embed_model
    )


def _run_git(repo: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_source_provenance(repo: Path) -> dict[str, Any] | None:
    """Return Git provenance only when the indexed path contains tracked files."""
    try:
        git_root = Path(_run_git(repo, ["rev-parse", "--show-toplevel"])).resolve()
        source_subdir = str(repo.relative_to(git_root))
        pathspec = source_subdir or "."
        tracked = _run_git(git_root, ["ls-files", "--", pathspec])
        if not tracked:
            return None
        commit = _run_git(git_root, ["rev-parse", "HEAD"])
        dirty = bool(
            _run_git(
                git_root,
                ["status", "--porcelain", "--untracked-files=all", "--", pathspec],
            )
        )
        try:
            remote = _run_git(git_root, ["config", "--get", "remote.origin.url"])
        except (OSError, subprocess.SubprocessError):
            remote = ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    location = remote or str(git_root)
    suffix = f":{source_subdir}" if source_subdir else ""
    return {
        "source_kind": "git",
        "source_path": str(repo),
        "source_commit": commit,
        "source_digest": "",
        "source_identity": f"git:{location}@{commit}{suffix}",
        "working_tree_dirty": dirty,
    }


def _directory_source_provenance(repo: Path, file_paths: set[str]) -> dict[str, Any]:
    """Hash indexed files when the source is not tracked by its own Git checkout."""
    digest = hashlib.sha256()
    for file_path in sorted(file_paths):
        absolute_path = repo / file_path
        if not absolute_path.is_file():
            continue
        digest.update(file_path.encode())
        digest.update(b"\0")
        digest.update(absolute_path.read_bytes())
        digest.update(b"\0")
    source_digest = digest.hexdigest()
    return {
        "source_kind": "directory",
        "source_path": str(repo),
        "source_commit": "",
        "source_digest": source_digest,
        "source_identity": f"sha256:{source_digest}",
        "working_tree_dirty": False,
    }


def _index_metadata(
    repo: Path,
    result: Any,
    embed_model: str,
    index_run_id: str,
    git_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    file_paths = {
        str(node.properties.get("file_path", ""))
        for node in result.nodes
        if node.properties.get("file_path")
    }
    source = git_provenance or _directory_source_provenance(repo, file_paths)
    return {
        **source,
        "last_indexed_commit": source["source_commit"],
        "index_run_id": index_run_id,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": embed_model,
    }


class IndexingListener:
    """Callback interface to track indexing pipeline progress and log messages."""

    def on_step(self, step_name: str, message: str, progress: float | None = None) -> None:
        """Called when a pipeline step makes progress.

        Args:
            step_name: The name of the pipeline stage (e.g., 'parse', 'llm_extract', 'resolution', 'db_write', 'embed_chunks', 'vector_write').
            message: A user-facing log message describing what is happening.
            progress: A float between 0.0 and 1.0 representing completeness of the step (or None if indeterminate).
        """
        pass


def _detect_git_changes(
    repo: Path,
    last_commit: str,
    languages: list[str],
    excludes: list[str] | None,
) -> tuple[set[str], set[str]] | None:
    """
    Detect modified (including added/untracked) and deleted files in a git repo.
    Returns (modified_files, deleted_files) or None if not a valid Git repo or commit.
    """
    import subprocess

    from hybrid_rag.ingestion.parser import LANGUAGE_BY_EXT

    def run_git(args: list[str]) -> str:
        res = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()

    try:
        # Check inside work tree
        is_inside = run_git(["rev-parse", "--is-inside-work-tree"])
        if is_inside != "true":
            return None

        # Verify if last_commit exists in git history
        run_git(["cat-file", "-t", last_commit])
    except Exception:
        # Not a git repo or commit doesn't exist
        return None

    try:
        try:
            git_root_str = run_git(["rev-parse", "--show-toplevel"])
            git_root = Path(git_root_str).resolve() if git_root_str else repo.resolve()
        except Exception:
            git_root = repo.resolve()
        repo_resolved = repo.resolve()

        # Get tracked changes relative to last_commit (includes working dir modifications)
        diff_out = run_git(["diff", "--name-status", last_commit, "--", str(repo_resolved)])
        # Get untracked files
        status_out = run_git(["status", "--porcelain", str(repo_resolved)])

        raw_modified = set()
        raw_deleted = set()

        for line in diff_out.splitlines():
            if not line:
                continue
            parts = line.split("\t")
            status = parts[0]
            if status.startswith("D"):
                raw_deleted.add(parts[1])
            elif status.startswith("R"):
                raw_deleted.add(parts[1])
                raw_modified.add(parts[2])
            else:
                raw_modified.add(parts[1])

        for line in status_out.splitlines():
            if line.startswith("?? "):
                raw_modified.add(line[3:].strip())

        def to_repo_rel(f_str: str) -> str | None:
            p = Path(f_str)
            if p.is_absolute():
                try:
                    return str(p.relative_to(repo_resolved))
                except ValueError:
                    return None
            if git_root == repo_resolved:
                return str(p)
            try:
                full = (git_root / p).resolve()
                return str(full.relative_to(repo_resolved))
            except ValueError:
                return None

        modified_files = {to_repo_rel(f) for f in raw_modified if to_repo_rel(f)}
        deleted_files = {to_repo_rel(f) for f in raw_deleted if to_repo_rel(f)}

        # Filtering logic
        import os

        exts = {ext for ext, lang in LANGUAGE_BY_EXT.items() if lang in languages}
        if os.environ.get("NON_CODE_INGESTION", "false").lower() == "true":
            exts.update({".yaml", ".yml", ".md", "Dockerfile"})

        exclude_set = (
            set(excludes)
            if excludes is not None
            else {
                ".venv",
                "venv",
                "experiments",
                "dist",
                "build",
                ".git",
                "__pycache__",
                "node_modules",
                ".agents",
                ".pytest_cache",
                ".ruff_cache",
                ".roo",
                ".clinerules",
            }
        )

        def is_valid(fpath_str: str) -> bool:
            path = Path(fpath_str)
            ext = path.suffix.lower()
            if path.name == "Dockerfile":
                ext = "Dockerfile"
            if ext not in exts:
                return False
            if any(p in exclude_set or p.startswith(".venv") for p in path.parts):
                return False
            return True

        valid_modified = {f for f in modified_files if is_valid(f)}
        valid_deleted = {f for f in deleted_files if is_valid(f)}
        return valid_modified, valid_deleted

    except Exception:
        return None


def run_indexing_pipeline(
    repo_path: Path,
    languages: list[str],
    repo_name: str,
    graph_store: GraphStore,
    vector_store: VectorStore,
    ollama_url: str,
    embed_model: str,
    llm_model: str,
    llm_extract: bool,
    max_tokens: int,
    listener: IndexingListener | None = None,
    excludes: list[str] | None = None,
    incremental: bool = False,
    rebuild: bool = False,
    from_commit: str | None = None,
) -> dict[str, Any]:
    """Parse a code repository and ingest its code graph and embeddings.

    Executes:
      1. AST parsing of the repository.
      2. LLM-assisted relationship extraction (optional).
      3. Local entity resolution.
      4. Global cross-repo entity resolution against FalkorDB.
      5. Graph ingestion into FalkorDB.
      6. Code chunking, embedding, and vector upsert into Qdrant.
    """
    t_start = time.perf_counter()
    listener = listener or IndexingListener()

    # 1. Resolve repository path
    repo = repo_path.resolve()
    if not repo.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo}")
    validate_local_provider_configuration()
    validate_local_ollama_url(ollama_url)
    _validate_python_repository(repo, languages, excludes)

    # Lazy imports to keep execution startups fast
    from hybrid_rag.ingestion.entity_resolver import namespace_unresolved_stubs, resolve, stub_count
    from hybrid_rag.ingestion.merger import merge_supplemental
    from hybrid_rag.ingestion.parser import ParseResult, parse_file, parse_repo
    from hybrid_rag.utils.tracing import start_span

    # ── 1. Parse AST ───────────────────────────────────────────────────────────
    is_incremental = False
    modified_files = None
    deleted_files = None
    git_provenance = _git_source_provenance(repo)
    head_commit = git_provenance["source_commit"] if git_provenance else None
    index_run_id = str(uuid.uuid4())

    previous_metadata = graph_store.get_repository_metadata(repo_name)
    incompatible_index = bool(
        isinstance(previous_metadata, dict)
        and previous_metadata
        and not _metadata_is_compatible(previous_metadata, embed_model)
    )
    if incompatible_index:
        listener.on_step(
            "parse",
            "Stored index schema or embedding model changed; performing a full replacement.",
            0.0,
        )

    if incremental and not rebuild and not incompatible_index:
        try:
            last_commit = from_commit or graph_store.get_repository_commit(repo_name)
            if head_commit and last_commit:
                git_changes = _detect_git_changes(repo, last_commit, languages, excludes)
                if git_changes is not None:
                    modified_files, deleted_files = git_changes
                    is_incremental = True
                    logger.info(
                        "Incremental sync detected: %d files modified/added, %d files deleted since %s",
                        len(modified_files),
                        len(deleted_files),
                        last_commit,
                    )
        except Exception as exc:
            logger.debug("Incremental check failed, falling back to full index: %s", exc)

    if is_incremental:
        listener.on_step(
            "parse",
            f"Incremental sync: {len(modified_files)} files modified, {len(deleted_files)} files deleted...",
            0.0,
        )

        if not modified_files and not deleted_files:
            metadata = _index_metadata(
                repo,
                ParseResult(),
                embed_model,
                index_run_id,
                git_provenance,
            )
            vector_store.set_repository_metadata(repo_name, metadata)
            graph_store.set_repository_metadata(repo_name, metadata)
            listener.on_step("complete", "Index is already up to date.", 1.0)
            return {
                "elapsed_seconds": time.perf_counter() - t_start,
                "nodes_parsed": 0,
                "edges_parsed": 0,
                "nodes_upserted": 0,
                "edges_upserted": 0,
                "chunks_embedded": 0,
                "vectors_upserted": 0,
            }

        # Parse only modified files
        result = ParseResult()
        for f in sorted(modified_files):
            file_res = parse_file(repo / f, repo, repo_name=repo_name)
            result.nodes.extend(file_res.nodes)
            result.edges.extend(file_res.edges)
            result.errors.extend(file_res.errors)

        listener.on_step(
            "parse",
            f"Incremental parse complete. Found {len(result.nodes)} nodes, {len(result.edges)} edges.",
            1.0,
        )
    else:
        # Full Ingest Parse AST
        listener.on_step(
            "parse",
            f"Parsing source files in {repo} for languages: {', '.join(languages)}...",
            None,
        )
        with start_span("pipeline_parse_ast", {"repo": str(repo)}):
            result = parse_repo(repo, languages=languages, repo_name=repo_name, excludes=excludes)
        listener.on_step(
            "parse",
            f"AST parsing complete. Found {len(result.nodes)} nodes, {len(result.edges)} edges, {len(result.errors)} errors.",
            1.0,
        )

    _ensure_repository_node(result, repo_name)

    metadata = _index_metadata(
        repo,
        result,
        embed_model,
        index_run_id,
        git_provenance,
    )

    # ── 2. LLM-assisted extraction (optional) ─────────────────────────────────
    if llm_extract:
        all_extra_edges: list = []
        if is_incremental:
            py_files = sorted(repo.rglob("*.py"))
            py_files = [fp for fp in py_files if str(fp.relative_to(repo)) in modified_files]
        else:
            py_files = sorted(repo.rglob("*.py"))
        total_files = len(py_files)

        listener.on_step(
            "llm_extract",
            f"Starting LLM-assisted extraction for {total_files} Python files...",
            0.0,
        )

        if total_files > 0:
            from hybrid_rag.ingestion.ollama_llm_extractor import OllamaLLMExtractor

            extractor_ctx = OllamaLLMExtractor(ollama_url=ollama_url, model=llm_model)

            with extractor_ctx as extractor:
                with start_span(
                    "pipeline_llm_extract", {"total_files": total_files, "model": llm_model}
                ):
                    for i, fp in enumerate(py_files, start=1):
                        try:
                            file_text = fp.read_text(encoding="utf-8", errors="replace")
                            file_result = parse_file(fp, repo, repo_name=repo_name)
                            extra = extractor.extract(file_text, file_result)
                            all_extra_edges.extend(extra)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("LLM extraction skipped %s: %s", fp, exc)

                        progress_val = float(i) / total_files
                        listener.on_step(
                            "llm_extract",
                            f"LLM extraction progress: processed {i}/{total_files} files ({len(all_extra_edges)} extra edges added).",
                            progress_val,
                        )

            result = merge_supplemental(result, all_extra_edges)

        listener.on_step(
            "llm_extract",
            f"LLM extraction complete. Total supplemental edges added: {len(all_extra_edges)}.",
            1.0,
        )
    else:
        listener.on_step("llm_extract", "LLM-assisted extraction skipped.", 1.0)

    # ── 3. Local Entity Resolution ─────────────────────────────────────────────
    listener.on_step("resolution", "Running local entity resolution...", 0.0)
    with start_span("pipeline_entity_resolution", {"type": "local"}):
        before_stubs = stub_count(result)
        result = resolve(result)
        after_stubs = stub_count(result)
        resolved = before_stubs - after_stubs
    listener.on_step(
        "resolution",
        f"Local entity resolution complete: merged {resolved} stub nodes, {after_stubs} external stubs remain.",
        0.5,
    )

    # ── 3b. Global Entity Resolution ───────────────────────────────────────────
    from hybrid_rag.ingestion.entity_resolver import resolve_global

    with start_span("pipeline_entity_resolution", {"type": "global"}):
        before_global_stubs = stub_count(result)
        result = resolve_global(result, graph_store)
        after_global_stubs = stub_count(result)
        global_resolved = before_global_stubs - after_global_stubs
    listener.on_step(
        "resolution",
        f"Global entity resolution complete: resolved {global_resolved} stubs against FalkorDB, {after_global_stubs} remain external.",
        1.0,
    )
    result = namespace_unresolved_stubs(result, repo_name)
    for node in result.nodes:
        node.properties["index_run_id"] = index_run_id
    for edge in result.edges:
        edge.properties["index_run_id"] = index_run_id

    # Generate every embedding before mutating the active index. Parsing,
    # extraction, and embedding failures therefore preserve the previous run.
    # ── 4. Chunk + Embed ───────────────────────────────────────────────────────
    listener.on_step("embed_chunks", "Chunking source files for vector indexing...", 0.0)

    from hybrid_rag.ingestion.chunker import chunk_file

    chunks_to_embed = []
    node_names = {
        node.id: node.properties.get("name", node.id.rsplit("::", 1)[-1]) for node in result.nodes
    }
    seen_files: set[str] = set()
    for node in result.nodes:
        fp = node.properties.get("file_path", "")
        if fp and fp not in seen_files:
            seen_files.add(fp)
            abs_fp = repo / fp
            file_chunks = chunk_file(abs_fp, repo, max_tokens=max_tokens, repo_name=repo_name)
            chunks_to_embed.extend(file_chunks)

    total_chunks = len(chunks_to_embed)
    vector_batches: list[list[dict[str, Any]]] = []

    listener.on_step(
        "embed_chunks",
        f"Found {total_chunks} chunks to embed. Generating embeddings...",
        0.0,
    )

    if total_chunks > 0:
        from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder

        embedder_ctx = OllamaEmbedder(ollama_url=ollama_url, model=embed_model)

        with embedder_ctx as embedder:
            with start_span(
                "pipeline_embed_chunks", {"total_chunks": total_chunks, "model": embed_model}
            ):
                batch_size = _embedding_batch_size()
                for i in range(0, total_chunks, batch_size):
                    batch = chunks_to_embed[i : i + batch_size]
                    batch_texts = [ch.text for ch in batch]
                    embeddings = embedder.embed_texts(batch_texts)

                    batch_payload = []
                    for ch, emb in zip(batch, embeddings):
                        label = ch.label
                        file_type = (
                            "non_code"
                            if label in ("Document", "Configuration", "Directory")
                            else "code"
                        )
                        batch_payload.append(
                            {
                                "node_id": f"{ch.node_id}::{ch.chunk_index}",
                                "name": node_names.get(ch.node_id, ch.node_id.rsplit("::", 1)[-1]),
                                "label": label,
                                "file_path": ch.file_path,
                                "text": ch.text,
                                "embedding": emb,
                                "repository": repo_name,
                                "file_type": file_type,
                                "indexed_commit": metadata["last_indexed_commit"],
                                "index_run_id": metadata["index_run_id"],
                                "source_identity": metadata["source_identity"],
                            }
                        )

                    vector_batches.append(batch_payload)

                    progress_val = float(min(i + len(batch), total_chunks)) / total_chunks
                    listener.on_step(
                        "embed_chunks",
                        f"Generated embeddings for {min(i + len(batch), total_chunks)}/{total_chunks} chunks.",
                        progress_val,
                    )

    # ── 5. Publish staged graph and vector data ────────────────────────────────
    listener.on_step("db_write", "Writing codebase graph data to FalkorDB...", 0.0)
    with start_span("pipeline_graph_write"):
        counts = graph_store.ingest(result)
    listener.on_step(
        "db_write",
        f"FalkorDB ingestion complete: upserted {counts['nodes']} nodes and {counts['edges']} edges.",
        1.0,
    )

    upserted = 0
    for batch_payload in vector_batches:
        upserted += vector_store.upsert(batch_payload)

    # New records are present before stale records are removed, avoiding an
    # empty-index window during rebuilds and incremental replacements.
    listener.on_step("cleanup", f"Removing stale {repo_name} index data...", 0.0)
    if is_incremental:
        for file_path in deleted_files | modified_files:
            graph_store.delete_file_nodes_except_run(file_path, repo_name, index_run_id)
            vector_store.delete_file_vectors_except_run(file_path, repo_name, index_run_id)
    else:
        graph_store.delete_repository_except_run(repo_name, index_run_id)
        vector_store.delete_repository_except_run(repo_name, index_run_id)
    listener.on_step("cleanup", f"Removed stale {repo_name} index data.", 1.0)

    # Trigger garbage collection
    import gc

    gc.collect()

    listener.on_step(
        "vector_write", f"Qdrant ingestion complete: upserted {upserted} vectors.", 1.0
    )

    # Incremental runs retain unchanged records, so publish one provenance
    # identity across the repository only after all writes and cleanup succeed.
    vector_store.set_repository_metadata(repo_name, metadata)
    graph_store.set_repository_metadata(repo_name, metadata)
    logger.info("Saved index provenance %s for %s", metadata["index_run_id"], repo_name)

    elapsed = time.perf_counter() - t_start
    listener.on_step(
        "complete", f"Indexing pipeline completed successfully in {elapsed:.2f}s.", 1.0
    )

    return {
        "elapsed_seconds": elapsed,
        "nodes_parsed": len(result.nodes),
        "edges_parsed": len(result.edges),
        "nodes_upserted": counts["nodes"],
        "edges_upserted": counts["edges"],
        "chunks_embedded": total_chunks,
        "vectors_upserted": upserted,
    }
