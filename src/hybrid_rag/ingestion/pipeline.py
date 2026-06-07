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
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.ingestion.parser import parse_file, parse_repo

    # ── 1. Parse AST ───────────────────────────────────────────────────────────
    listener.on_step(
        "parse", f"Parsing source files in {repo} for languages: {', '.join(languages)}...", None
    )
    result = parse_repo(repo, languages=languages, repo_name=repo_name, excludes=excludes)
    listener.on_step(
        "parse",
        f"AST parsing complete. Found {len(result.nodes)} nodes, {len(result.edges)} edges, {len(result.errors)} errors.",
        1.0,
    )

    # ── 2. LLM-assisted extraction (optional) ─────────────────────────────────
    if llm_extract:
        from hybrid_rag.ingestion.ollama_llm_extractor import OllamaLLMExtractor

        all_extra_edges: list = []
        py_files = sorted(repo.rglob("*.py"))
        total_files = len(py_files)

        listener.on_step(
            "llm_extract",
            f"Starting LLM-assisted extraction for {total_files} Python files...",
            0.0,
        )

        if total_files > 0:
            with OllamaLLMExtractor(ollama_url=ollama_url, model=llm_model) as extractor:
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
    counts = graph_store.ingest(result)
    listener.on_step(
        "db_write",
        f"FalkorDB ingestion complete: upserted {counts['nodes']} nodes and {counts['edges']} edges.",
        1.0,
    )

    # ── 5. Chunk + Embed + Vector Ingest ───────────────────────────────────────
    listener.on_step("embed_chunks", "Chunking source files for vector indexing...", 0.0)

    source_lines: dict[str, list[str]] = {}
    for node in result.nodes:
        fp = node.properties.get("file_path", "")
        if fp and fp not in source_lines:
            abs_fp = repo / fp
            try:
                source_lines[fp] = abs_fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                source_lines[fp] = []

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
    all_chunks: list[dict] = []

    listener.on_step(
        "embed_chunks",
        f"Found {total_chunks} chunks to embed. Generating embeddings via Ollama...",
        0.0,
    )

    if total_chunks > 0:
        with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
            batch_size = 128
            for i in range(0, total_chunks, batch_size):
                batch = chunks_to_embed[i : i + batch_size]
                batch_texts = [ch.text for ch in batch]
                embeddings = embedder.embed_texts(batch_texts)
                for ch, emb in zip(batch, embeddings):
                    all_chunks.append(
                        {
                            "node_id": f"{ch.node_id}::{ch.chunk_index}",
                            "label": ch.label,
                            "file_path": ch.file_path,
                            "text": ch.text,
                            "embedding": emb,
                            "repository": repo_name,
                        }
                    )

                progress_val = float(min(i + len(batch), total_chunks)) / total_chunks
                listener.on_step(
                    "embed_chunks",
                    f"Generated embeddings for {min(i + len(batch), total_chunks)}/{total_chunks} chunks.",
                    progress_val,
                )

    listener.on_step("vector_write", f"Upserting {len(all_chunks)} chunks to Qdrant...", 0.0)
    upserted = vector_store.upsert(all_chunks)
    listener.on_step(
        "vector_write", f"Qdrant ingestion complete: upserted {upserted} vectors.", 1.0
    )

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
