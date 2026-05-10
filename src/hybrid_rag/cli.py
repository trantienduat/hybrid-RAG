"""
CLI entry point for hybrid-rag.

Commands:
  hybrid-rag index REPO   — parse repo, write KG + vectors
  hybrid-rag status       — check services health

Usage:
  hybrid-rag index ./path/to/repo
  hybrid-rag index ./path/to/repo --languages python java
  hybrid-rag status
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(name="hybrid-rag", help="Privacy-preserving Graph-Hybrid RAG for codebases.")
console = Console()
err_console = Console(stderr=True, style="bold red")


# ── index command ──────────────────────────────────────────────────────────────

@app.command()
def index(
    repo: Path = typer.Argument(..., help="Path to repository root to index."),
    languages: list[str] = typer.Option(
        ["python"], "--languages", "-l",
        help="Source languages to parse (python, java).",
    ),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option("nomic-embed-text", envvar="EMBED_MODEL"),
    max_tokens: int = typer.Option(512, help="Max tokens per chunk."),
) -> None:
    """Parse REPO and ingest code graph + embeddings into FalkorDB and Qdrant."""
    repo = repo.resolve()
    if not repo.is_dir():
        err_console.print(f"[ERROR] Not a directory: {repo}")
        raise typer.Exit(1)

    console.rule(f"[bold cyan]hybrid-rag index[/] — {repo}")
    t0 = time.perf_counter()

    # Lazy imports so CLI is fast to load.
    # Composition root: business logic is typed against ports; concrete adapters
    # are wired here so the rest of the codebase stays vendor-neutral.
    from hybrid_rag.ingestion.parser import parse_repo
    from hybrid_rag.ingestion.entity_resolver import resolve, stub_count
    from hybrid_rag.ingestion.chunker import chunk_nodes
    from hybrid_rag.ports import GraphStore, VectorStore, BaseEmbedder
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.vector.qdrant_store import QdrantStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder

    # ── 1. Parse ───────────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("Parsing source files…", total=None)
        result = parse_repo(repo, languages=languages)
        progress.update(task, description=f"Parsed — {len(result.nodes)} nodes, "
                                          f"{len(result.edges)} edges, "
                                          f"{len(result.errors)} errors")

    if result.errors:
        console.print(f"[yellow]Parse warnings:[/] {len(result.errors)}")
        for err in result.errors[:5]:
            console.print(f"  • {err}")
        if len(result.errors) > 5:
            console.print(f"  … and {len(result.errors) - 5} more")

    # ── 2. Entity resolution ───────────────────────────────────────────────────
    before_stubs = stub_count(result)
    result = resolve(result)
    after_stubs = stub_count(result)
    resolved = before_stubs - after_stubs
    console.print(f"Entity resolver: {resolved} stubs merged → {after_stubs} external stubs remain")

    # ── 3. Graph ingest ────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("Writing to FalkorDB…", total=None)
        graph_store: GraphStore = FalkorDBStore(
            host=graph_host, port=graph_port, graph_name=graph_name
        )
        counts = graph_store.ingest(result)
        progress.update(task, description=f"FalkorDB — {counts['nodes']} nodes, "
                                          f"{counts['edges']} edges upserted")

    console.print(f"[green]✓[/] Graph: {counts['nodes']} nodes, {counts['edges']} edges")

    # ── 4. Chunk + embed + vector upsert ───────────────────────────────────────
    try:
        source_lines: dict[str, list[str]] = {}
        for node in result.nodes:
            fp = node.properties.get("file_path", "")
            if fp and fp not in source_lines:
                abs_fp = repo / fp
                try:
                    source_lines[fp] = abs_fp.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    source_lines[fp] = []

        chunks = chunk_nodes(result.nodes, [], max_tokens=max_tokens)
        # chunk_nodes needs per-file lines; do it per-file instead
        from hybrid_rag.ingestion.chunker import chunk_file
        all_chunks: list[dict] = []

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                      console=console) as progress:
            task = progress.add_task("Embedding chunks…", total=None)
            embedder: BaseEmbedder
            with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
                # Build chunks per file then embed
                seen_files: set[str] = set()
                for node in result.nodes:
                    fp = node.properties.get("file_path", "")
                    if fp and fp not in seen_files:
                        seen_files.add(fp)
                        abs_fp = repo / fp
                        file_chunks = chunk_file(abs_fp, repo, max_tokens=max_tokens)
                        for ch in file_chunks:
                            emb = embedder.embed_query(ch.text)
                            all_chunks.append({
                                "node_id": f"{ch.node_id}::{ch.chunk_index}",
                                "label": ch.label,
                                "file_path": ch.file_path,
                                "text": ch.text,
                                "embedding": emb,
                            })
                progress.update(task, description=f"Embedded {len(all_chunks)} chunks")

        vector_store: VectorStore = QdrantStore(
            host=qdrant_host, port=qdrant_port, collection=qdrant_collection
        )
        upserted = vector_store.upsert(all_chunks)
        console.print(f"[green]✓[/] Vectors: {upserted} chunks upserted to Qdrant")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Embedding/vector stage failed: {exc}")
        raise typer.Exit(1) from exc

    elapsed = time.perf_counter() - t0
    console.rule(f"[bold green]Done in {elapsed:.1f}s[/]")


# ── status command ─────────────────────────────────────────────────────────────

@app.command()
def status(
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
) -> None:
    """Check health of FalkorDB, Qdrant, and Ollama services."""
    import httpx

    ok = True

    # FalkorDB
    try:
        from hybrid_rag.graph.falkordb_store import FalkorDBStore
        gc = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        n = gc.node_count()
        console.print(f"[green]✓[/] FalkorDB  {graph_host}:{graph_port}/{graph_name}  — {n} nodes")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/] FalkorDB  — {exc}")
        ok = False

    # Qdrant
    try:
        from hybrid_rag.vector.qdrant_store import QdrantStore
        vc = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)
        n = vc.point_count()
        console.print(f"[green]✓[/] Qdrant    {qdrant_host}:{qdrant_port}/{qdrant_collection}  — {n} points")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/] Qdrant    — {exc}")
        ok = False

    # Ollama
    try:
        resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        console.print(f"[green]✓[/] Ollama    {ollama_url}  — models: {', '.join(models) or 'none'}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/] Ollama    — {exc}")
        ok = False

    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
