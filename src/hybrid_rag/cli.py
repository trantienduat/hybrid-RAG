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

import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(name="hybrid-rag", help="Privacy-preserving Graph-Hybrid RAG for codebases.")
console = Console()
err_console = Console(stderr=True, style="bold red")
logger = logging.getLogger(__name__)


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
    llm_model: str = typer.Option("qwen2.5-coder:7b", envvar="LLM_MODEL"),
    llm_extract: bool = typer.Option(
        False, "--llm-extract/--no-llm-extract",
        help="Run LLM-assisted extraction to supplement AST edges (slower, more complete).",
    ),
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
    from hybrid_rag.ingestion.parser import parse_repo, parse_file
    from hybrid_rag.ingestion.entity_resolver import resolve, stub_count
    from hybrid_rag.ingestion.merger import merge_supplemental
    from hybrid_rag.ingestion.chunker import chunk_nodes
    from hybrid_rag.ports import GraphStore, VectorStore, BaseEmbedder, BaseLLMExtractor
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

    # ── 2. LLM-assisted extraction (optional) ─────────────────────────────────
    if llm_extract:
        from hybrid_rag.ingestion.ollama_llm_extractor import OllamaLLMExtractor
        all_extra_edges: list = []
        py_files = sorted(repo.rglob("*.py"))
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                      console=console) as progress:
            task = progress.add_task(f"LLM extraction (0/{len(py_files)} files)…", total=None)
            with OllamaLLMExtractor(ollama_url=ollama_url, model=llm_model) as extractor:
                for i, fp in enumerate(py_files, start=1):
                    try:
                        file_text = fp.read_text(encoding="utf-8", errors="replace")
                        file_result = parse_file(fp, repo)
                        extra = extractor.extract(file_text, file_result)
                        all_extra_edges.extend(extra)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("LLM extraction skipped %s: %s", fp, exc)
                    progress.update(task, description=f"LLM extraction ({i}/{len(py_files)} files)…")
        result = merge_supplemental(result, all_extra_edges)
        console.print(f"[green]✓[/] LLM extraction: {len(all_extra_edges)} supplemental edges added")

    # ── 3. Entity resolution ───────────────────────────────────────────────────
    before_stubs = stub_count(result)
    result = resolve(result)
    after_stubs = stub_count(result)
    resolved = before_stubs - after_stubs
    console.print(f"Entity resolver: {resolved} stubs merged → {after_stubs} external stubs remain")

    # ── 4. Graph ingest ────────────────────────────────────────────────────────
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

    # ── 5. Chunk + embed + vector upsert ───────────────────────────────────────
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


# ── query command ──────────────────────────────────────────────────────────────

@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question about the codebase."),
    top_k: int = typer.Option(10, help="Total candidates to retrieve before context assembly."),
    context_n: int = typer.Option(5, help="Top results to include in the assembled context."),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option("nomic-embed-text", envvar="EMBED_MODEL"),
    rrf_k: int = typer.Option(60, help="RRF k parameter (default: 60)."),
) -> None:
    """Query the indexed codebase using hybrid graph + vector retrieval."""
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
    from hybrid_rag.retrieval.query_analyzer import analyze
    from hybrid_rag.vector.qdrant_store import QdrantStore

    console.rule("[bold cyan]hybrid-rag query[/]")

    analysis = analyze(question)
    console.print(
        f"[dim]Query type:[/] [bold]{analysis.query_type}[/]  "
        f"entities: {analysis.entities}  keywords: {analysis.keywords[:5]}"
    )

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)

        with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
            retriever = HybridRetriever(
                graph_store=graph_store,
                vector_store=vector_store,
                embedder=embedder,
                rrf_k=rrf_k,
            )
            ctx = retriever.retrieve_with_context(question, top_k=top_k, context_n=context_n)

        console.print(
            f"\n[bold]Retrieved {ctx.metadata['total_results']} results "
            f"(showing top {ctx.metadata['shown']})[/]"
        )
        console.print(
            f"Sources: graph={ctx.metadata['has_graph']}, "
            f"vector={ctx.metadata['has_vector']}\n"
        )
        console.rule("[dim]Context[/]")
        console.print(ctx.text)
        console.rule()

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Query failed: {exc}")
        raise typer.Exit(1) from exc


# ── eval command ───────────────────────────────────────────────────────────────

@app.command()
def eval(
    queries: list[str] = typer.Option(
        [], "--query", "-q",
        help="Run only these query IDs (e.g. Q4 Q9). Defaults to full Q1-Q20 corpus.",
    ),
    top_k: int = typer.Option(10, help="Candidates to retrieve per query."),
    rrf_k: int = typer.Option(60, help="RRF k parameter."),
    rrf_structural_weight: float = typer.Option(3.0, help="Graph weight for structural queries."),
    rrf_hybrid_weight: float = typer.Option(1.5, help="Graph weight for hybrid queries."),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option("nomic-embed-text", envvar="EMBED_MODEL"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON results to stdout."),
) -> None:
    """Run M3 baseline evaluation: hybrid vs vector-only on Q1-Q20."""
    import json as _json

    from rich.table import Table

    from hybrid_rag.eval.corpus import EVAL_CORPUS
    from hybrid_rag.eval.runner import EvalRunner
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.vector.qdrant_store import QdrantStore

    # Filter corpus if caller specified specific IDs
    corpus = EVAL_CORPUS
    if queries:
        ids = {q.upper() for q in queries}
        corpus = [c for c in EVAL_CORPUS if c.id.upper() in ids]
        if not corpus:
            err_console.print(f"[ERROR] No matching query IDs found: {queries}")
            raise typer.Exit(1)

    console.rule("[bold cyan]hybrid-rag eval[/]")
    console.print(
        f"Corpus: {len(corpus)} queries  |  top_k={top_k}  "
        f"rrf_k={rrf_k}  w_structural={rrf_structural_weight}  w_hybrid={rrf_hybrid_weight}"
    )

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)

        with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
            runner = EvalRunner(
                graph_store=graph_store,
                vector_store=vector_store,
                embedder=embedder,
                rrf_k=rrf_k,
                rrf_structural_weight=rrf_structural_weight,
                rrf_hybrid_weight=rrf_hybrid_weight,
            )
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Running {len(corpus)} queries…", total=None)
                report = runner.run(corpus, top_k=top_k)
                progress.update(task, description="Done")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Eval failed: {exc}")
        raise typer.Exit(1) from exc

    if json_out:
        rows = []
        for r in report.results:
            rows.append({
                "id": r.query_id, "hops": r.hops, "query_type": r.query_type,
                "gt_count": len(r.ground_truth),
                "hit5_hybrid": r.hit_at5_hybrid, "hit5_vector": r.hit_at5_vector,
                "mrr_hybrid": r.mrr_hybrid, "mrr_vector": r.mrr_vector,
                "delta_hit": r.delta_hit,
            })
        console.print(_json.dumps(rows, indent=2))
        return

    # ── Rich results table ────────────────────────────────────────────────────
    console.print()
    tbl = Table(title="Q1-Q20 Evaluation Results", show_lines=True)
    tbl.add_column("ID",    style="bold", width=4)
    tbl.add_column("Hops",  justify="center", width=5)
    tbl.add_column("Type",  width=11)
    tbl.add_column("GT",    justify="right", width=4)
    tbl.add_column("Hit@5 Hybrid", justify="center", width=13)
    tbl.add_column("Hit@5 Vector", justify="center", width=13)
    tbl.add_column("Δ",     justify="center", width=6)
    tbl.add_column("MRR Hybrid", justify="center", width=11)
    tbl.add_column("MRR Vector", justify="center", width=11)

    for r in report.results:
        gt_str = str(len(r.ground_truth)) if r.ground_truth else "[dim]0[/]"
        h_hit  = f"[green]{r.hit_at5_hybrid:.1f}[/]" if r.hit_at5_hybrid >= 0.5 else f"[red]{r.hit_at5_hybrid:.1f}[/]"
        v_hit  = f"[green]{r.hit_at5_vector:.1f}[/]" if r.hit_at5_vector >= 0.5 else f"[red]{r.hit_at5_vector:.1f}[/]"
        delta  = (
            f"[bold green]+{r.delta_hit:.1f}[/]" if r.delta_hit > 0
            else (f"[red]{r.delta_hit:.1f}[/]" if r.delta_hit < 0 else f"[dim]{r.delta_hit:.1f}[/]")
        )
        tbl.add_row(
            r.query_id, str(r.hops), r.query_type, gt_str,
            h_hit, v_hit, delta,
            f"{r.mrr_hybrid:.2f}", f"{r.mrr_vector:.2f}",
        )

    console.print(tbl)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    summary = Table(title="Summary", show_lines=False, box=None)
    summary.add_column("Metric", style="bold", width=40)
    summary.add_column("Hybrid", justify="right", width=10)
    summary.add_column("Vector-only", justify="right", width=12)
    summary.add_column("Δ", justify="right", width=8)

    def _row(label: str, h: float, v: float) -> None:
        d = h - v
        d_str = f"[bold green]+{d:.3f}[/]" if d > 0 else (f"[red]{d:.3f}[/]" if d < 0 else f"{d:.3f}")
        summary.add_row(label, f"{h:.3f}", f"{v:.3f}", d_str)

    _row("Hit@5 (all)",          report.hit5_hybrid_all,  report.hit5_vector_all)
    _row("Hit@5 (1-hop)",        report.hit5_hybrid_by_hops[1], report.hit5_vector_by_hops[1])
    _row("Hit@5 (2-hop)",        report.hit5_hybrid_by_hops[2], report.hit5_vector_by_hops[2])
    _row("Hit@5 (3-hop)",        report.hit5_hybrid_by_hops[3], report.hit5_vector_by_hops[3])
    _row("MRR  (all)",           report.mrr_hybrid_all,   report.mrr_vector_all)
    _row("ΔHitRate 2-3hop",      report.delta_hit_2_3hop + report.hit5_vector_all,
         report.hit5_vector_all)  # formatted separately below

    # Override last row with cleaner format
    console.print(summary)
    console.print()

    target_met = report.delta_hit_2_3hop >= 0.20
    delta_color = "bold green" if target_met else "yellow"
    console.print(
        f"[{delta_color}]ΔHitRate 2-3hop = {report.delta_hit_2_3hop:+.3f}[/]  "
        f"(thesis target: ≥ +0.20  {'✓ MET' if target_met else '✗ not met yet'})"
    )


# ── eval command ───────────────────────────────────────────────────────────────

@app.command()
def eval(
    queries: list[str] = typer.Option(
        [], "--query", "-q",
        help="Run only these query IDs (e.g. Q4 Q9). Defaults to full Q1-Q20 corpus.",
    ),
    top_k: int = typer.Option(10, help="Candidates to retrieve per query."),
    rrf_k: int = typer.Option(60, help="RRF k parameter."),
    rrf_structural_weight: float = typer.Option(3.0, help="Graph weight for structural queries."),
    rrf_hybrid_weight: float = typer.Option(1.5, help="Graph weight for hybrid queries."),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option("nomic-embed-text", envvar="EMBED_MODEL"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON results to stdout."),
) -> None:
    """Run M3 baseline evaluation: hybrid vs vector-only on Q1-Q20."""
    import json as _json

    from rich.table import Table

    from hybrid_rag.eval.corpus import EVAL_CORPUS
    from hybrid_rag.eval.runner import EvalRunner
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.vector.qdrant_store import QdrantStore

    # Filter corpus if caller specified specific IDs
    corpus = EVAL_CORPUS
    if queries:
        ids = {q.upper() for q in queries}
        corpus = [c for c in EVAL_CORPUS if c.id.upper() in ids]
        if not corpus:
            err_console.print(f"[ERROR] No matching query IDs found: {queries}")
            raise typer.Exit(1)

    console.rule("[bold cyan]hybrid-rag eval[/]")
    console.print(
        f"Corpus: {len(corpus)} queries  |  top_k={top_k}  "
        f"rrf_k={rrf_k}  w_structural={rrf_structural_weight}  w_hybrid={rrf_hybrid_weight}"
    )

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)

        with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
            runner = EvalRunner(
                graph_store=graph_store,
                vector_store=vector_store,
                embedder=embedder,
                rrf_k=rrf_k,
                rrf_structural_weight=rrf_structural_weight,
                rrf_hybrid_weight=rrf_hybrid_weight,
            )
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Running {len(corpus)} queries…", total=None)
                report = runner.run(corpus, top_k=top_k)
                progress.update(task, description="Done")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Eval failed: {exc}")
        raise typer.Exit(1) from exc

    if json_out:
        rows = []
        for r in report.results:
            rows.append({
                "id": r.query_id, "hops": r.hops, "query_type": r.query_type,
                "gt_count": len(r.ground_truth),
                "hit5_hybrid": r.hit_at5_hybrid, "hit5_vector": r.hit_at5_vector,
                "mrr_hybrid": r.mrr_hybrid, "mrr_vector": r.mrr_vector,
                "delta_hit": r.delta_hit,
            })
        console.print(_json.dumps(rows, indent=2))
        return

    # ── Rich results table ────────────────────────────────────────────────────
    console.print()
    tbl = Table(title="Q1-Q20 Evaluation Results", show_lines=True)
    tbl.add_column("ID",    style="bold", width=4)
    tbl.add_column("Hops",  justify="center", width=5)
    tbl.add_column("Type",  width=11)
    tbl.add_column("GT",    justify="right", width=4)
    tbl.add_column("Hit@5 Hybrid", justify="center", width=13)
    tbl.add_column("Hit@5 Vector", justify="center", width=13)
    tbl.add_column("Δ",     justify="center", width=6)
    tbl.add_column("MRR Hybrid", justify="center", width=11)
    tbl.add_column("MRR Vector", justify="center", width=11)

    for r in report.results:
        gt_str = str(len(r.ground_truth)) if r.ground_truth else "[dim]0[/]"
        h_hit  = f"[green]{r.hit_at5_hybrid:.1f}[/]" if r.hit_at5_hybrid >= 0.5 else f"[red]{r.hit_at5_hybrid:.1f}[/]"
        v_hit  = f"[green]{r.hit_at5_vector:.1f}[/]" if r.hit_at5_vector >= 0.5 else f"[red]{r.hit_at5_vector:.1f}[/]"
        delta  = (
            f"[bold green]+{r.delta_hit:.1f}[/]" if r.delta_hit > 0
            else (f"[red]{r.delta_hit:.1f}[/]" if r.delta_hit < 0 else f"[dim]{r.delta_hit:.1f}[/]")
        )
        tbl.add_row(
            r.query_id, str(r.hops), r.query_type, gt_str,
            h_hit, v_hit, delta,
            f"{r.mrr_hybrid:.2f}", f"{r.mrr_vector:.2f}",
        )

    console.print(tbl)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    summary = Table(title="Summary", show_lines=False, box=None)
    summary.add_column("Metric", style="bold", width=40)
    summary.add_column("Hybrid", justify="right", width=10)
    summary.add_column("Vector-only", justify="right", width=12)
    summary.add_column("Δ", justify="right", width=8)

    def _row(label: str, h: float, v: float) -> None:
        d = h - v
        d_str = f"[bold green]+{d:.3f}[/]" if d > 0 else (f"[red]{d:.3f}[/]" if d < 0 else f"{d:.3f}")
        summary.add_row(label, f"{h:.3f}", f"{v:.3f}", d_str)

    _row("Hit@5 (all)",          report.hit5_hybrid_all,  report.hit5_vector_all)
    _row("Hit@5 (1-hop)",        report.hit5_hybrid_by_hops[1], report.hit5_vector_by_hops[1])
    _row("Hit@5 (2-hop)",        report.hit5_hybrid_by_hops[2], report.hit5_vector_by_hops[2])
    _row("Hit@5 (3-hop)",        report.hit5_hybrid_by_hops[3], report.hit5_vector_by_hops[3])
    _row("MRR  (all)",           report.mrr_hybrid_all,   report.mrr_vector_all)
    _row("ΔHitRate 2-3hop",      report.delta_hit_2_3hop + report.hit5_vector_all,
         report.hit5_vector_all)  # formatted separately below

    # Override last row with cleaner format
    console.print(summary)
    console.print()

    target_met = report.delta_hit_2_3hop >= 0.20
    delta_color = "bold green" if target_met else "yellow"
    console.print(
        f"[{delta_color}]ΔHitRate 2-3hop = {report.delta_hit_2_3hop:+.3f}[/]  "
        f"(thesis target: ≥ +0.20  {'✓ MET' if target_met else '✗ not met yet'})"
    )


if __name__ == "__main__":
    app()
