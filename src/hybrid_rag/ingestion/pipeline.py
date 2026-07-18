"""
Unified indexing pipeline for Hybrid-RAG.

Allows running the parsing, extraction, resolution, and ingestion steps
programmatically with progress callbacks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from hybrid_rag.ports import GraphStore, VectorStore

logger = logging.getLogger(__name__)


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
        # Get tracked changes relative to last_commit (includes working dir modifications)
        diff_out = run_git(["diff", "--name-status", last_commit])
        # Get untracked files
        status_out = run_git(["status", "--porcelain"])

        modified_files = set()
        deleted_files = set()

        for line in diff_out.splitlines():
            if not line:
                continue
            parts = line.split("\t")
            status = parts[0]
            if status.startswith("D"):
                deleted_files.add(parts[1])
            elif status.startswith("R"):
                deleted_files.add(parts[1])
                modified_files.add(parts[2])
            else:
                modified_files.add(parts[1])

        for line in status_out.splitlines():
            if line.startswith("?? "):
                modified_files.add(line[3:])

        # Filtering logic
        exts = {ext for ext, lang in LANGUAGE_BY_EXT.items() if lang in languages}
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
            }
        )

        def is_valid(fpath_str: str) -> bool:
            path = Path(fpath_str)
            if path.suffix.lower() not in exts:
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

    # Lazy imports to keep execution startups fast
    from hybrid_rag.ingestion.entity_resolver import resolve, stub_count
    from hybrid_rag.ingestion.merger import merge_supplemental
    from hybrid_rag.ingestion.parser import ParseResult, parse_file, parse_repo
    from hybrid_rag.utils.tracing import start_span

    # ── 1. Parse AST ───────────────────────────────────────────────────────────
    is_incremental = False
    modified_files = None
    deleted_files = None
    head_commit = None

    if incremental and not rebuild:
        try:
            import subprocess

            res_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
            )
            head_commit = res_head.stdout.strip()

            last_commit = from_commit or graph_store.get_repository_commit(repo_name)
            if last_commit:
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

        # Cleanup deleted & modified files from stores to avoid duplicates/orphans
        for f in deleted_files | modified_files:
            graph_store.delete_file_nodes(f, repo_name)
            vector_store.delete_file_vectors(f, repo_name)

        if not modified_files and not deleted_files:
            listener.on_step("complete", "Index is already up to date.", 1.0)
            if head_commit:
                graph_store.set_repository_commit(repo_name, head_commit)
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

        # Get HEAD commit for full ingest so we can do incremental sync next time
        try:
            import subprocess

            res_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
            )
            head_commit = res_head.stdout.strip()
        except Exception:
            pass

    # ── 2. LLM-assisted extraction (optional) ─────────────────────────────────
    if llm_extract:
        import os

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
            is_gemini = llm_model.startswith("gemini") or os.environ.get("LLM_PROVIDER") == "gemini"
            if is_gemini:
                from hybrid_rag.ingestion.gemini_llm_extractor import GeminiLLMExtractor

                extractor_ctx = GeminiLLMExtractor(model=llm_model)
            else:
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

    # ── 4. Graph Ingest ────────────────────────────────────────────────────────
    listener.on_step("db_write", "Writing codebase graph data to FalkorDB...", 0.0)
    with start_span("pipeline_graph_write"):
        counts = graph_store.ingest(result)
    listener.on_step(
        "db_write",
        f"FalkorDB ingestion complete: upserted {counts['nodes']} nodes and {counts['edges']} edges.",
        1.0,
    )

    # ── 5. Chunk + Embed + Vector Ingest ───────────────────────────────────────
    listener.on_step("embed_chunks", "Chunking source files for vector indexing...", 0.0)

    from hybrid_rag.ingestion.chunker import chunk_file

    chunks_to_embed = []
    seen_files: set[str] = set()
    for node in result.nodes:
        fp = node.properties.get("file_path", "")
        if fp and fp not in seen_files:
            seen_files.add(fp)
            abs_fp = repo / fp
            file_chunks = chunk_file(abs_fp, repo, max_tokens=max_tokens)
            chunks_to_embed.extend(file_chunks)

    total_chunks = len(chunks_to_embed)
    upserted = 0

    listener.on_step(
        "embed_chunks",
        f"Found {total_chunks} chunks to embed. Generating embeddings...",
        0.0,
    )

    if total_chunks > 0:
        import os

        is_gemini_embed = (
            embed_model.startswith("gemini")
            or embed_model == "text-embedding-004"
            or os.environ.get("EMBED_PROVIDER") == "gemini"
        )
        if is_gemini_embed:
            from hybrid_rag.ingestion.gemini_embedder import GeminiEmbedder

            embedder_ctx = GeminiEmbedder(model=embed_model)
        else:
            from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder

            embedder_ctx = OllamaEmbedder(ollama_url=ollama_url, model=embed_model)

        with embedder_ctx as embedder:
            with start_span(
                "pipeline_embed_chunks", {"total_chunks": total_chunks, "model": embed_model}
            ):
                batch_size = 128
                for i in range(0, total_chunks, batch_size):
                    batch = chunks_to_embed[i : i + batch_size]
                    batch_texts = [ch.text for ch in batch]
                    embeddings = embedder.embed_texts(batch_texts)

                    batch_payload = []
                    for ch, emb in zip(batch, embeddings):
                        batch_payload.append(
                            {
                                "node_id": f"{ch.node_id}::{ch.chunk_index}",
                                "label": ch.label,
                                "file_path": ch.file_path,
                                "text": ch.text,
                                "embedding": emb,
                                "repository": repo_name,
                            }
                        )

                    # Upsert each batch immediately to free RAM
                    upserted += vector_store.upsert(batch_payload)

                    # Clear variables to allow garbage collection
                    batch_payload = None
                    embeddings = None
                    batch_texts = None

                    progress_val = float(min(i + len(batch), total_chunks)) / total_chunks
                    listener.on_step(
                        "embed_chunks",
                        f"Generated and upserted embeddings for {min(i + len(batch), total_chunks)}/{total_chunks} chunks.",
                        progress_val,
                    )

    # Trigger garbage collection
    import gc

    gc.collect()

    listener.on_step(
        "vector_write", f"Qdrant ingestion complete: upserted {upserted} vectors.", 1.0
    )

    # Save HEAD commit state to FalkorDB for next incremental sync
    if head_commit:
        try:
            graph_store.set_repository_commit(repo_name, head_commit)
            logger.info("Saved last indexed commit %s to graph store metadata", head_commit)
        except Exception as exc:
            logger.debug("Failed to save commit state to graph store: %s", exc)

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
