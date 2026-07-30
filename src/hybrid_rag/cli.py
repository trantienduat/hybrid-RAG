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
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from hybrid_rag.config import app_config
from hybrid_rag.constants import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL

app = typer.Typer(name="hybrid-rag", help="Privacy-preserving Graph-Hybrid RAG for codebases.")
console = Console()
err_console = Console(stderr=True, style="bold red")
logger = logging.getLogger(__name__)


# ── index command ──────────────────────────────────────────────────────────────


@app.command()
def index(
    repo: Path = typer.Argument(..., help="Path to repository root to index."),
    languages: list[str] = typer.Option(
        ["python"],
        "--languages",
        "-l",
        help="Source languages to parse (python, java).",
    ),
    repo_name: str = typer.Option(None, help="Custom namespace name for the repository."),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, envvar="EMBED_MODEL"),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, envvar="LLM_MODEL"),
    llm_extract: bool = typer.Option(
        False,
        "--llm-extract/--no-llm-extract",
        help="Run LLM-assisted extraction to supplement AST edges (slower, more complete).",
    ),
    max_tokens: int = typer.Option(512, help="Max tokens per chunk."),
    exclude: list[str] = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Folder or file name patterns to exclude from parsing (multi-value allowed).",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--no-incremental",
        help="Run indexing incrementally based on Git changes since last indexed commit.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Force a full rebuild and overwrite cached data.",
    ),
    from_commit: str = typer.Option(
        None,
        "--from-commit",
        help="Commit hash to compare against for incremental sync (overrides auto-detection).",
    ),
) -> None:
    """Parse REPO and ingest code graph + embeddings into FalkorDB and Qdrant."""
    repo = repo.resolve()
    if not repo.is_dir():
        err_console.print(f"[ERROR] Not a directory: {repo}")
        raise typer.Exit(1)

    repo_namespace = repo_name or repo.name
    console.rule(f"[bold cyan]hybrid-rag index[/] — {repo} [dim](namespace: {repo_namespace})[/]")
    t0 = time.perf_counter()

    # Lazy imports so CLI is fast to load.
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.pipeline import IndexingListener, run_indexing_pipeline
    from hybrid_rag.vector.qdrant_store import QdrantStore

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            class CliIndexingListener(IndexingListener):
                def __init__(self) -> None:
                    self.current_task = None
                    self.last_step = None

                def on_step(
                    self, step_name: str, message: str, progress_val: float | None = None
                ) -> None:
                    if self.last_step != step_name:
                        if self.current_task is not None:
                            progress.update(self.current_task, completed=100)
                        self.current_task = progress.add_task(
                            message, total=100 if progress_val is not None else None
                        )
                        self.last_step = step_name
                    else:
                        if progress_val is not None:
                            progress.update(
                                self.current_task,
                                description=message,
                                completed=int(progress_val * 100),
                            )
                        else:
                            progress.update(self.current_task, description=message)

                    if progress_val == 1.0:
                        progress.update(
                            self.current_task, description=f"[green]✓[/] {message}", completed=100
                        )

            listener = CliIndexingListener()
            graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
            vector_store = QdrantStore(
                host=qdrant_host, port=qdrant_port, collection=qdrant_collection
            )

            run_indexing_pipeline(
                repo_path=repo,
                languages=languages,
                repo_name=repo_namespace,
                graph_store=graph_store,
                vector_store=vector_store,
                ollama_url=ollama_url,
                embed_model=embed_model,
                llm_model=llm_model,
                llm_extract=llm_extract,
                max_tokens=max_tokens,
                listener=listener,
                excludes=exclude,
                incremental=incremental,
                rebuild=rebuild,
                from_commit=from_commit,
            )

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Indexing pipeline failed: {exc}")
        raise typer.Exit(1) from exc

    elapsed = time.perf_counter() - t0
    console.rule(f"[bold green]Done in {elapsed:.1f}s[/]")


# ── community-build command ───────────────────────────────────────────────────


@app.command("community-build")
def community_build(
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, envvar="LLM_MODEL"),
) -> None:
    """Run community clustering and compile architectural summaries using FalkorDB and Ollama."""
    console.rule("[bold cyan]hybrid-rag community-build[/]")
    t0 = time.perf_counter()

    from hybrid_rag.graph.community_builder import CommunityBuilder
    from hybrid_rag.graph.falkordb_store import FalkorDBStore

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console
        ) as progress:
            task = progress.add_task(
                "Running community partitioning and summarization…", total=None
            )
            builder = CommunityBuilder(
                graph_store=graph_store,
                ollama_url=ollama_url,
                llm_model=llm_model,
            )
            count = builder.build_communities()
            progress.update(task, description=f"Done — compiled {count} communities!")

        console.print(
            f"[green]✓[/] Communities: {count} architectural modules generated and persisted to FalkorDB."
        )
    except Exception as exc:
        err_console.print(f"[ERROR] Community build failed: {exc}")
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
        console.print(
            f"[green]✓[/] Qdrant    {qdrant_host}:{qdrant_port}/{qdrant_collection}  — {n} points"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/] Qdrant    — {exc}")
        ok = False

    # Ollama
    try:
        resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        console.print(
            f"[green]✓[/] Ollama    {ollama_url}  — models: {', '.join(models) or 'none'}"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/] Ollama    — {exc}")
        ok = False

    if not ok:
        raise typer.Exit(1)


# ── query command ──────────────────────────────────────────────────────────────


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question about the codebase."),
    top_k: int = typer.Option(20, help="Total candidates to retrieve before context assembly."),
    context_n: int = typer.Option(5, help="Legacy top results limit fallback."),
    max_tokens: int = typer.Option(None, help="Maximum tokens for dynamic context budget."),
    max_chars: int = typer.Option(None, help="Maximum characters for dynamic context budget."),
    repo_name: str = typer.Option(
        None, help="Scope query search to a specific repository namespace."
    ),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, envvar="EMBED_MODEL"),
    rrf_k: int = typer.Option(60, help="RRF k parameter (default: 60)."),
    rrf_graph_weight: float = typer.Option(
        app_config.rrf_structural_weight,
        help="Graph-list weight for local RRF fusion.",
    ),
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
                rrf_structural_weight=rrf_graph_weight,
            )
            # Route dynamic vs legacy parameters with repository scoping
            if max_tokens is None and max_chars is None:
                ctx = retriever.retrieve_with_context(
                    question,
                    top_k=top_k,
                    max_tokens=None,
                    max_chars=None,
                    context_n=context_n,
                    repository=repo_name,
                )
            else:
                ctx = retriever.retrieve_with_context(
                    question,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    max_chars=max_chars,
                    repository=repo_name,
                )

        if ctx.metadata.get("use_budget"):
            console.print(
                f"\n[bold green]✓[/] [bold]Retrieved {ctx.metadata['total_results']} candidates | "
                f"Packed {ctx.metadata['shown']} chunks into prompt context[/]"
            )
            console.print(
                f"  • Budget Utilized: [bold cyan]{ctx.metadata['total_tokens']}/{ctx.metadata['budget_limit']}[/] estimated tokens "
                f"({ctx.metadata['total_chars']} characters)"
            )
            console.print(
                f"  • Chunks: [green]{ctx.metadata['chunks_included_count']} included[/] | "
                f"[yellow]{ctx.metadata['chunks_excluded_count']} excluded (due to budget limit)[/]"
            )
        else:
            console.print(
                f"\n[bold]Retrieved {ctx.metadata['total_results']} results "
                f"(showing top {ctx.metadata['shown']})[/]"
            )
        console.print(
            f"  • Sources: graph={ctx.metadata['has_graph']}, vector={ctx.metadata['has_vector']}\n"
        )
        console.rule("[dim]Context[/]")
        console.print(ctx.text)
        console.rule()

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Query failed: {exc}")
        raise typer.Exit(1) from exc


# [eval command moved to unified implementation below]


# ── eval command ───────────────────────────────────────────────────────────────


@app.command()
def eval(
    queries: list[str] = typer.Option(
        [],
        "--query",
        "-q",
        help="Run only these query IDs (e.g. Q4 Q29). Defaults to full Q1-Q50 corpus (diagnostic mode only).",
    ),
    mode: str = typer.Option(
        "diagnostic",
        "--mode",
        help="Evaluation mode: 'diagnostic' (default structural/hybrid Q1-Q50) or 'repoqa' (Searching Needle Function).",
    ),
    benchmark_path: Path = typer.Option(
        None,
        "--benchmark-path",
        help="Path to the custom RepoQA benchmark JSON file (required for repoqa mode).",
    ),
    top_k: int = typer.Option(10, help="Candidates to retrieve per query."),
    rrf_k: int = typer.Option(60, help="RRF k parameter."),
    rrf_structural_weight: float = typer.Option(1.5, help="Unified graph weight for RRF fusion."),
    rrf_hybrid_weight: float = typer.Option(1.5, help="Graph weight for hybrid queries (unused)."),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, envvar="EMBED_MODEL"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON results to stdout."),
    repo_name: str = typer.Option(
        None,
        "--repo-name",
        help="Scope evaluation queries to a specific repository namespace.",
    ),
) -> None:
    """Run codebase evaluation (diagnostic structural/hybrid or RepoQA Searching Needle Function)."""
    import json as _json

    from rich.table import Table

    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.vector.qdrant_store import QdrantStore

    if not repo_name:
        err_console.print("[ERROR] --repo-name is required for reproducible evaluation.")
        raise typer.Exit(1)

    if mode.lower() == "repoqa":
        if not benchmark_path:
            err_console.print("[ERROR] --benchmark-path is required in 'repoqa' evaluation mode.")
            raise typer.Exit(1)

        from hybrid_rag.eval.corpus import load_repoqa_json
        from hybrid_rag.eval.runner import RepoQAEvalRunner

        try:
            corpus = load_repoqa_json(benchmark_path)
        except Exception as exc:
            err_console.print(f"[ERROR] Failed to load benchmark JSON: {exc}")
            raise typer.Exit(1)

        console.rule("[bold cyan]hybrid-rag eval (repoqa mode)[/]")
        console.print(
            f"Benchmark: {benchmark_path}  |  Queries: {len(corpus)}  |  "
            f"top_k={top_k}  rrf_k={rrf_k}"
        )

        try:
            graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
            vector_store = QdrantStore(
                host=qdrant_host, port=qdrant_port, collection=qdrant_collection
            )
            from hybrid_rag.eval.preflight import validate_index_provenance

            validate_index_provenance(graph_store, vector_store, repo_name)

            with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
                runner = RepoQAEvalRunner(
                    graph_store=graph_store,
                    vector_store=vector_store,
                    embedder=embedder,
                    rrf_k=rrf_k,
                    rrf_structural_weight=rrf_structural_weight,
                    rrf_hybrid_weight=rrf_hybrid_weight,
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Running {len(corpus)} RepoQA queries…", total=None)
                    report = runner.run(corpus, top_k=top_k, repository=repo_name)
                    progress.update(task, description="Done")

        except Exception as exc:
            err_console.print(f"[ERROR] RepoQA Eval failed: {exc}")
            raise typer.Exit(1) from exc

        if json_out:
            rows = []
            for r in report.results:
                rows.append(
                    {
                        "id": r.query_id,
                        "target_function": r.target_function,
                        "file_path": r.file_path,
                        "rank_hybrid": r.rank_hybrid,
                        "rank_vector": r.rank_vector,
                        "hit1_hybrid": r.hit_at1_hybrid,
                        "hit1_vector": r.hit_at1_vector,
                        "hit5_hybrid": r.hit_at5_hybrid,
                        "hit5_vector": r.hit_at5_vector,
                        "mrr_hybrid": r.mrr_hybrid,
                        "mrr_vector": r.mrr_vector,
                    }
                )
            console.print(_json.dumps(rows, indent=2))
            return

        # Rich RepoQA table
        console.print()
        tbl = Table(title="RepoQA Evaluation Results", show_lines=True)
        tbl.add_column("ID", style="bold", width=15)
        tbl.add_column("Target Function", width=20)
        tbl.add_column("File Path", width=25)
        tbl.add_column("Rank Hybrid", justify="center", width=12)
        tbl.add_column("Rank Vector", justify="center", width=12)
        tbl.add_column("MRR Hybrid", justify="center", width=11)
        tbl.add_column("MRR Vector", justify="center", width=11)

        for r in report.results:
            rank_h_str = (
                f"[green]{r.rank_hybrid}[/]"
                if 0 < r.rank_hybrid <= 5
                else (f"[red]{r.rank_hybrid}[/]" if r.rank_hybrid > 5 else "[dim]Not Found[/]")
            )
            rank_v_str = (
                f"[green]{r.rank_vector}[/]"
                if 0 < r.rank_vector <= 5
                else (f"[red]{r.rank_vector}[/]" if r.rank_vector > 5 else "[dim]Not Found[/]")
            )
            tbl.add_row(
                r.query_id,
                r.target_function,
                r.file_path,
                rank_h_str,
                rank_v_str,
                f"{r.mrr_hybrid:.2f}",
                f"{r.mrr_vector:.2f}",
            )
        console.print(tbl)

        # Summary Table
        console.print()
        summary = Table(title="RepoQA Summary", show_lines=False, box=None)
        summary.add_column("Metric", style="bold", width=40)
        summary.add_column("Hybrid", justify="right", width=10)
        summary.add_column("Vector-only", justify="right", width=12)
        summary.add_column("Δ", justify="right", width=8)

        def _row(label: str, h: float, v: float) -> None:
            d = h - v
            d_str = (
                f"[bold green]+{d:.3f}[/]"
                if d > 0
                else (f"[red]{d:.3f}[/]" if d < 0 else f"{d:.3f}")
            )
            summary.add_row(label, f"{h:.3f}", f"{v:.3f}", d_str)

        _row("Hit@1", report.hit1_hybrid, report.hit1_vector)
        _row("Hit@5", report.hit5_hybrid, report.hit5_vector)
        _row("MRR", report.mrr_hybrid, report.mrr_vector)

        console.print(summary)
        console.print()
        return

    # Default 'diagnostic' mode
    from hybrid_rag.eval.corpus import EVAL_CORPUS
    from hybrid_rag.eval.runner import EvalRunner

    corpus = EVAL_CORPUS
    if queries:
        ids = {q.upper() for q in queries}
        corpus = [c for c in EVAL_CORPUS if c.id.upper() in ids]
        if not corpus:
            err_console.print(f"[ERROR] No matching query IDs found: {queries}")
            raise typer.Exit(1)

    console.rule("[bold cyan]hybrid-rag eval (diagnostic mode)[/]")
    console.print(
        f"Corpus: {len(corpus)} queries  |  top_k={top_k}  "
        f"rrf_k={rrf_k}  w_structural={rrf_structural_weight}  w_hybrid={rrf_hybrid_weight}"
    )

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)
        from hybrid_rag.eval.preflight import validate_index_provenance

        validate_index_provenance(graph_store, vector_store, repo_name)

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
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Running {len(corpus)} queries…", total=None)
                report = runner.run(corpus, top_k=top_k, repository=repo_name)
                progress.update(task, description="Done")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Eval failed: {exc}")
        raise typer.Exit(1) from exc

    if json_out:
        rows = []
        for r in report.results:
            rows.append(
                {
                    "id": r.query_id,
                    "hops": r.hops,
                    "query_type": r.query_type,
                    "gt_count": len(r.ground_truth),
                    "hit5_hybrid": r.hit_at5_hybrid,
                    "hit5_vector": r.hit_at5_vector,
                    "mrr_hybrid": r.mrr_hybrid,
                    "mrr_vector": r.mrr_vector,
                    "delta_hit": r.delta_hit,
                }
            )
        console.print(_json.dumps(rows, indent=2))
        return

    # ── Rich results table ────────────────────────────────────────────────────
    console.print()
    tbl = Table(title=f"Q1-Q{len(corpus)} Evaluation Results", show_lines=True)
    tbl.add_column("ID", style="bold", width=4)
    tbl.add_column("Hops", justify="center", width=5)
    tbl.add_column("Type", width=11)
    tbl.add_column("GT", justify="right", width=4)
    tbl.add_column("Hit@5 Hybrid", justify="center", width=13)
    tbl.add_column("Hit@5 Vector", justify="center", width=13)
    tbl.add_column("Δ", justify="center", width=6)
    tbl.add_column("MRR Hybrid", justify="center", width=11)
    tbl.add_column("MRR Vector", justify="center", width=11)

    for r in report.results:
        gt_str = str(len(r.ground_truth)) if r.ground_truth else "[dim]0[/]"
        h_hit = (
            f"[green]{r.hit_at5_hybrid:.1f}[/]"
            if r.hit_at5_hybrid >= 0.5
            else f"[red]{r.hit_at5_hybrid:.1f}[/]"
        )
        v_hit = (
            f"[green]{r.hit_at5_vector:.1f}[/]"
            if r.hit_at5_vector >= 0.5
            else f"[red]{r.hit_at5_vector:.1f}[/]"
        )
        delta = (
            f"[bold green]+{r.delta_hit:.1f}[/]"
            if r.delta_hit > 0
            else (f"[red]{r.delta_hit:.1f}[/]" if r.delta_hit < 0 else f"[dim]{r.delta_hit:.1f}[/]")
        )
        tbl.add_row(
            r.query_id,
            str(r.hops),
            r.query_type,
            gt_str,
            h_hit,
            v_hit,
            delta,
            f"{r.mrr_hybrid:.2f}",
            f"{r.mrr_vector:.2f}",
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
        d_str = (
            f"[bold green]+{d:.3f}[/]" if d > 0 else (f"[red]{d:.3f}[/]" if d < 0 else f"{d:.3f}")
        )
        summary.add_row(label, f"{h:.3f}", f"{v:.3f}", d_str)

    _row("Hit@5 (all)", report.hit5_hybrid_all, report.hit5_vector_all)
    _row("Hit@5 (1-hop)", report.hit5_hybrid_by_hops[1], report.hit5_vector_by_hops[1])
    _row("Hit@5 (2-hop)", report.hit5_hybrid_by_hops[2], report.hit5_vector_by_hops[2])
    _row("Hit@5 (3-hop)", report.hit5_hybrid_by_hops[3], report.hit5_vector_by_hops[3])
    _row("MRR  (all)", report.mrr_hybrid_all, report.mrr_vector_all)
    _row(
        "ΔHitRate 2-3hop", report.delta_hit_2_3hop + report.hit5_vector_all, report.hit5_vector_all
    )  # formatted separately below

    console.print(summary)
    console.print()

    target_met = report.delta_hit_2_3hop >= 0.20
    delta_color = "bold green" if target_met else "yellow"
    console.print(
        f"[{delta_color}]ΔHitRate 2-3hop = {report.delta_hit_2_3hop:+.3f}[/]  "
        f"(thesis target: ≥ +0.20  {'✓ MET' if target_met else '✗ not met yet'})"
    )


# ── serve command ──────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)."),
    workers: int = typer.Option(1, help="Number of worker processes (ignored with --reload)."),
    log_level: str = typer.Option("info", help="Uvicorn log level."),
) -> None:
    """Start the hybrid-rag FastAPI server (M4 #27)."""
    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        err_console.print("[ERROR] uvicorn not installed. Run: pip install 'hybrid-rag[api]'")
        raise typer.Exit(1)

    console.rule("[bold cyan]hybrid-rag serve[/]")
    console.print(f"  API:    http://{host}:{port}")
    console.print(f"  UI:     http://{host}:{port}/")
    console.print(f"  Docs:   http://{host}:{port}/docs")
    console.rule()

    uvicorn.run(
        "hybrid_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level=log_level,
    )


# ── ragas command ──────────────────────────────────────────────────────────────


@app.command()
def ragas(
    top_k: int = typer.Option(20, help="Retrieval candidates per query."),
    context_n: int = typer.Option(5, help="Context chunks assembled for LLM."),
    llm_model: str = typer.Option(DEFAULT_LLM_MODEL, envvar="LLM_MODEL"),
    judge_model: str = typer.Option(
        None,
        envvar="RAGAS_JUDGE_MODEL",
        help="Local model used to judge answers; defaults to --llm-model.",
    ),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, envvar="EMBED_MODEL"),
    repo_name: str = typer.Option(
        None,
        "--repo-name",
        help="Repository namespace to evaluate (required).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON to stdout."),
    subset: str = typer.Option("all", help="Corpus subset: all | 1hop | 2hop | 3hop | hybrid"),
) -> None:
    """Run RAGAS generation-quality evaluation (M4 #31)."""
    import json as _json

    from rich.table import Table

    from hybrid_rag.eval.corpus import EVAL_CORPUS, ONE_HOP, THREE_HOP, TWO_HOP
    from hybrid_rag.eval.corpus import HYBRID as HYBRID_CASES
    from hybrid_rag.eval.ground_truth import build_reference_answer, compute_ground_truth
    from hybrid_rag.eval.preflight import require_complete_ground_truth, validate_index_provenance
    from hybrid_rag.eval.ragas_runner import RagasRunner
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
    from hybrid_rag.vector.qdrant_store import QdrantStore

    subsets = {
        "all": EVAL_CORPUS,
        "1hop": ONE_HOP,
        "2hop": TWO_HOP,
        "3hop": THREE_HOP,
        "hybrid": HYBRID_CASES,
    }
    corpus = subsets.get(subset, EVAL_CORPUS)

    if not repo_name:
        err_console.print("[ERROR] --repo-name is required for reproducible RAGAS evaluation.")
        raise typer.Exit(1)

    console.rule("[bold cyan]hybrid-rag ragas[/]")
    console.print(f"Corpus: {len(corpus)} queries  |  llm={llm_model}  top_k={top_k}")

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)
        validate_index_provenance(graph_store, vector_store, repo_name)
        ground_truth = {
            case.id: compute_ground_truth(graph_store, case, repo_name) for case in corpus
        }
        require_complete_ground_truth(ground_truth)
        references = {
            case.id: build_reference_answer(case, ground_truth[case.id]) for case in corpus
        }

        with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
            retriever = HybridRetriever(
                graph_store=graph_store,
                vector_store=vector_store,
                embedder=embedder,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Generating + scoring {len(corpus)} queries…", total=None)
                runner = RagasRunner(
                    retriever=retriever,
                    repository=repo_name,
                    ollama_url=ollama_url,
                    llm_model=llm_model,
                    judge_model=judge_model,
                    embedding_model=embed_model,
                )
                report = runner.run(
                    corpus,
                    references=references,
                    top_k=top_k,
                    context_n=context_n,
                )
                progress.update(task, description="Done")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] RAGAS eval failed: {exc}")
        raise typer.Exit(1) from exc

    if json_out:
        console.print(_json.dumps(report.as_dict(), indent=2))
        return

    # Rich table
    tbl = Table(title="RAGAS Results", show_lines=True)
    tbl.add_column("ID", width=4)
    tbl.add_column("Hops", justify="center", width=5)
    tbl.add_column("Type", width=11)
    tbl.add_column("Faithfulness", justify="center", width=14)
    tbl.add_column("Ans. Relevancy", justify="center", width=15)
    tbl.add_column("Ctx. Precision", justify="center", width=15)
    tbl.add_column("Latency (ms)", justify="right", width=13)

    for s in report.samples:

        def _fmt(v: float | None) -> str:
            if v is None:
                return "[dim]n/a[/]"
            colour = "green" if v >= 0.7 else ("yellow" if v >= 0.4 else "red")
            return f"[{colour}]{v:.3f}[/]"

        tbl.add_row(
            s.query_id,
            str(s.hops),
            s.query_type,
            _fmt(s.faithfulness),
            _fmt(s.answer_relevancy),
            _fmt(s.context_precision),
            f"{s.latency_ms:.0f}",
        )

    console.print(tbl)
    console.print()

    d = report.as_dict()
    context_precision = d.get("context_precision")
    context_precision_text = f"{context_precision:.3f}" if context_precision is not None else "n/a"
    console.print(
        f"[bold]Overall[/]  faithfulness={d['faithfulness']:.3f}  "
        f"answer_relevancy={d['answer_relevancy']:.3f}  "
        f"context_precision={context_precision_text}  "
        f"avg_latency={d['avg_latency_ms']:.0f}ms"
    )
    console.rule()


# ── bench command ──────────────────────────────────────────────────────────────


@app.command()
def bench(
    n_runs: int = typer.Option(5, help="Timed runs per query."),
    top_k: int = typer.Option(20, help="Retrieval candidates (match production config)."),
    corpus: bool = typer.Option(
        False, "--corpus", help="Use Q1-Q50 corpus instead of default queries."
    ),
    graph_host: str = typer.Option("localhost", envvar="FALKORDB_HOST"),
    graph_port: int = typer.Option(6379, envvar="FALKORDB_PORT"),
    graph_name: str = typer.Option("codebase", envvar="FALKORDB_GRAPH"),
    qdrant_host: str = typer.Option("localhost", envvar="QDRANT_HOST"),
    qdrant_port: int = typer.Option(6333, envvar="QDRANT_PORT"),
    qdrant_collection: str = typer.Option("code_chunks", envvar="QDRANT_COLLECTION"),
    ollama_url: str = typer.Option("http://localhost:11434", envvar="OLLAMA_BASE_URL"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, envvar="EMBED_MODEL"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON to stdout."),
) -> None:
    """Measure p50/p95/p99 retrieval latency per query type (M4 #32)."""
    import json as _json

    from rich.table import Table

    from hybrid_rag.eval.benchmark import BenchmarkRunner
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
    from hybrid_rag.vector.qdrant_store import QdrantStore

    console.rule("[bold cyan]hybrid-rag bench[/]")
    console.print(f"n_runs={n_runs}  top_k={top_k}  corpus={corpus}")

    try:
        graph_store = FalkorDBStore(host=graph_host, port=graph_port, graph_name=graph_name)
        vector_store = QdrantStore(host=qdrant_host, port=qdrant_port, collection=qdrant_collection)

        with OllamaEmbedder(ollama_url=ollama_url, model=embed_model) as embedder:
            retriever = HybridRetriever(
                graph_store=graph_store,
                vector_store=vector_store,
                embedder=embedder,
            )
            runner = BenchmarkRunner(retriever)
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Running benchmark…", total=None)
                if corpus:
                    report = runner.run_from_corpus(n_runs=n_runs, top_k=top_k)
                else:
                    report = runner.run(n_runs=n_runs, top_k=top_k)
                progress.update(task, description="Done")

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[ERROR] Benchmark failed: {exc}")
        raise typer.Exit(1) from exc

    if json_out:
        console.print(_json.dumps(report.as_dict(), indent=2))
        return

    tbl = Table(title="Retrieval Latency Benchmark", show_lines=True)
    tbl.add_column("Query Type", width=13)
    tbl.add_column("n", justify="right", width=5)
    tbl.add_column("p50 (ms)", justify="right", width=9)
    tbl.add_column("p95 (ms)", justify="right", width=9)
    tbl.add_column("p99 (ms)", justify="right", width=9)
    tbl.add_column("mean (ms)", justify="right", width=10)
    tbl.add_column("min (ms)", justify="right", width=9)
    tbl.add_column("max (ms)", justify="right", width=9)

    for s in report.stats:

        def _c(v: float) -> str:
            colour = "green" if v < 500 else ("yellow" if v < 1500 else "red")
            return f"[{colour}]{v:.1f}[/]"

        tbl.add_row(
            s.query_type,
            str(s.n_runs),
            _c(s.p50),
            _c(s.p95),
            _c(s.p99),
            _c(s.mean),
            f"{s.min:.1f}",
            f"{s.max:.1f}",
        )

    console.print(tbl)
    console.rule()


@app.command()
def mcp(
    transport: str = typer.Option("stdio", help="Transport mode: stdio or sse."),
    host: str = typer.Option("localhost", help="Bind host for SSE transport."),
    port: int = typer.Option(8001, help="Bind port for SSE transport."),
) -> None:
    """Start the Model Context Protocol (MCP) server for local AI agents."""
    from hybrid_rag.mcp.server import mcp as mcp_server

    if transport == "sse":
        console.rule("[bold cyan]hybrid-rag mcp[/]")
        console.print(f"  Transport: {transport}")
        console.print(f"  SSE URL:   http://{host}:{port}/sse")
        console.print(f"  Messages:  http://{host}:{port}/messages")
        console.rule()
    else:
        logger.info("Starting Hybrid-RAG MCP server in stdio mode...")

    if transport == "stdio":
        mcp_server.run(transport="stdio")
    elif transport == "sse":
        mcp_server.settings.host = host
        mcp_server.settings.port = port
        mcp_server.run(transport="sse")
    else:
        err_console.print(f"[ERROR] Invalid transport: {transport}. Must be 'stdio' or 'sse'.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
