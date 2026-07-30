"""
Full evaluation pipeline: Hit Rate + RAGAS + Latency benchmarks.

Runs all 50 queries from the eval corpus against the live Hybrid-RAG system,
measures hit rate per hop-category, RAGAS quality scores, and latency breakdown.

Outputs:
  - results/evaluation_results.json  (machine-readable)
  - results/evaluation_report.md     (human-readable English report)

Usage:
    python scripts/run_full_evaluation.py [--repo llama-core] [--top-k 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid_rag.constants import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL
from hybrid_rag.eval.corpus import (
    EVAL_CORPUS,
    QueryCase,
)
from hybrid_rag.eval.ground_truth import build_reference_answer, compute_ground_truth
from hybrid_rag.eval.preflight import require_complete_ground_truth, validate_index_provenance
from hybrid_rag.eval.runner import _extract_names
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.vector.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Hit Rate Evaluation ───────────────────────────────────────────────────────


def evaluate_hit_rate(
    retriever: HybridRetriever,
    graph: FalkorDBStore,
    cases: list[QueryCase],
    top_k: int = 20,
    hit_at_k: int = 5,
    repository: str | None = None,
) -> list[dict]:
    """Evaluate Hit Rate @K for each query case."""
    results = []
    for case in cases:
        if not repository:
            raise ValueError("repository is required for evaluation")
        gt_names = compute_ground_truth(graph, case, repository)
        if not gt_names:
            logger.warning("%s: empty ground truth, skipping", case.id)
            results.append(
                {
                    "id": case.id,
                    "hops": case.hops,
                    "query_type": case.query_type,
                    "question": case.question,
                    "hit": False,
                    "gt_count": 0,
                    "retrieved_names": [],
                    "latency_ms": 0,
                    "error": "empty ground truth",
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            retrieved = retriever.retrieve(case.question, top_k=top_k, repository=repository)
        except Exception as exc:
            logger.warning("Retrieval failed for %s: %s", case.id, exc)
            results.append(
                {
                    "id": case.id,
                    "hops": case.hops,
                    "query_type": case.query_type,
                    "question": case.question,
                    "hit": False,
                    "gt_count": len(gt_names),
                    "retrieved_names": [],
                    "latency_ms": 0,
                    "error": str(exc),
                }
            )
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        # Check top-K for any match with ground truth
        top_results = retrieved[:hit_at_k]
        retrieved_names = [name.lower() for name in _extract_names(top_results)]

        hit = bool(gt_names & set(retrieved_names))

        results.append(
            {
                "id": case.id,
                "hops": case.hops,
                "query_type": case.query_type,
                "question": case.question,
                "hit": hit,
                "gt_count": len(gt_names),
                "gt_names": sorted(gt_names)[:10],  # cap for readability
                "retrieved_names": retrieved_names,
                "latency_ms": round(latency_ms, 2),
            }
        )
        logger.info(
            "%s [%d-hop] hit=%s  latency=%.0fms  gt=%d  retrieved=%d",
            case.id,
            case.hops,
            hit,
            latency_ms,
            len(gt_names),
            len(retrieved_names),
        )

    return results


# ── Vector-only Baseline ──────────────────────────────────────────────────────


def evaluate_vector_only(
    vector_store: QdrantStore,
    graph: FalkorDBStore,
    cases: list[QueryCase],
    top_k: int = 20,
    hit_at_k: int = 5,
    ollama_url: str = "http://localhost:11434",
    repository: str | None = None,
) -> list[dict]:
    """Evaluate Hit Rate using vector-only retrieval (no graph)."""
    import httpx

    results = []
    for case in cases:
        if not repository:
            raise ValueError("repository is required for evaluation")
        gt_names = compute_ground_truth(graph, case, repository)
        if not gt_names:
            results.append(
                {
                    "id": case.id,
                    "hops": case.hops,
                    "hit": False,
                    "gt_count": 0,
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            # Embed query
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": case.question},
                )
                resp.raise_for_status()
                query_vector = resp.json()["embedding"]

            # Vector search only
            filter_payload = {"repository": repository} if repository else None
            vector_results = vector_store.search(
                query_vector,
                top_k=top_k,
                filter_payload=filter_payload,
            )
        except Exception as exc:
            logger.warning("Vector-only retrieval failed for %s: %s", case.id, exc)
            results.append(
                {
                    "id": case.id,
                    "hops": case.hops,
                    "hit": False,
                    "gt_count": len(gt_names),
                    "error": str(exc),
                }
            )
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_names = [name.lower() for name in _extract_names(vector_results[:hit_at_k])]

        hit = bool(gt_names & set(retrieved_names))
        results.append(
            {
                "id": case.id,
                "hops": case.hops,
                "query_type": case.query_type,
                "hit": hit,
                "gt_count": len(gt_names),
                "retrieved_names": retrieved_names,
                "latency_ms": round(latency_ms, 2),
            }
        )
        logger.info("VECTOR-ONLY %s hit=%s latency=%.0fms", case.id, hit, latency_ms)

    return results


# ── Report Generator ──────────────────────────────────────────────────────────


def compute_hit_rate(results: list[dict], label: str = "") -> dict:
    """Compute hit rates grouped by hop count."""
    by_hops: dict[int, list[bool]] = defaultdict(list)
    valid_results = [r for r in results if r.get("gt_count", 0) > 0]
    for r in valid_results:
        by_hops[r["hops"]].append(r["hit"])

    all_hits = [r["hit"] for r in valid_results]
    summary = {
        "label": label,
        "skipped_empty_ground_truth": len(results) - len(valid_results),
        "overall": {
            "hit_rate": sum(all_hits) / len(all_hits) if all_hits else 0,
            "hits": sum(all_hits),
            "total": len(all_hits),
        },
    }
    for hops in sorted(by_hops.keys()):
        hits = by_hops[hops]
        summary[f"{hops}-hop"] = {
            "hit_rate": sum(hits) / len(hits) if hits else 0,
            "hits": sum(hits),
            "total": len(hits),
        }
    return summary


def generate_report(
    hybrid_results: list[dict],
    vector_results: list[dict],
    latency_data: list[dict],
    ragas_data: dict | None,
    timestamp: str,
) -> str:
    """Generate a human-readable Markdown report in English."""
    hybrid_summary = compute_hit_rate(hybrid_results, "Hybrid-RAG")
    vector_summary = compute_hit_rate(vector_results, "Vector-only RAG")
    valid_queries = hybrid_summary["overall"]["total"]
    total_queries = len(hybrid_results)

    lines = [
        "# Hybrid-RAG Evaluation Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Dataset:** LlamaIndex core ({len(hybrid_results)} queries)",
        f"**Ground-truth coverage:** {valid_queries}/{total_queries} "
        f"({'valid' if valid_queries == total_queries else 'incomplete'})",
        "**Hardware:** Apple Mac Studio (M2 Max, 64GB RAM)",
        "**Config:** `top_k=20, rrf_k=60, structural_weight=3.0, hybrid_weight=1.5`",
        "",
        "---",
        "",
        "## 1. Hit Rate @5 Comparison",
        "",
        "| Hop Category | Vector-only RAG | Hybrid-RAG | Delta (Δ) |",
        "|---|---|---|---|",
    ]

    for key in ["1-hop", "2-hop", "3-hop"]:
        if key in hybrid_summary and key in vector_summary:
            h_rate = hybrid_summary[key]["hit_rate"]
            v_rate = vector_summary[key]["hit_rate"]
            delta = h_rate - v_rate
            h_detail = f"{hybrid_summary[key]['hits']}/{hybrid_summary[key]['total']}"
            v_detail = f"{vector_summary[key]['hits']}/{vector_summary[key]['total']}"
            lines.append(
                f"| {key} | {v_rate:.3f} ({v_detail}) | {h_rate:.3f} ({h_detail}) | {delta:+.3f} |"
            )

    h_overall = hybrid_summary["overall"]
    v_overall = vector_summary["overall"]
    delta_overall = h_overall["hit_rate"] - v_overall["hit_rate"]
    lines.append(
        f"| **Overall (micro-avg)** | **{v_overall['hit_rate']:.3f}** ({v_overall['hits']}/{v_overall['total']}) "
        f"| **{h_overall['hit_rate']:.3f}** ({h_overall['hits']}/{h_overall['total']}) "
        f"| **{delta_overall:+.3f}** |"
    )

    # Per-query detail
    lines += [
        "",
        "### Per-Query Detail",
        "",
        "| Query | Hops | Type | Vector Hit | Hybrid Hit | Hybrid Latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for h, v in zip(hybrid_results, vector_results):
        v_hit = "✅" if v.get("hit") else "❌"
        h_hit = "✅" if h.get("hit") else "❌"
        lat = h.get("latency_ms", 0)
        lines.append(
            f"| {h['id']} | {h['hops']} | {h.get('query_type', '')} | {v_hit} | {h_hit} | {lat:.0f} |"
        )

    # Latency breakdown
    if latency_data:
        lines += [
            "",
            "---",
            "",
            "## 2. Latency Breakdown",
            "",
            "Average retrieval latency per query type:",
            "",
            "| Query Type | Avg Latency (ms) | Min (ms) | Max (ms) |",
            "|---|---|---|---|",
        ]
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in hybrid_results:
            if r.get("latency_ms"):
                by_type[r.get("query_type", "unknown")].append(r["latency_ms"])

        for qt, lats in sorted(by_type.items()):
            avg = sum(lats) / len(lats) if lats else 0
            mn = min(lats) if lats else 0
            mx = max(lats) if lats else 0
            lines.append(f"| {qt} | {avg:.0f} | {mn:.0f} | {mx:.0f} |")

        all_lats = [r["latency_ms"] for r in hybrid_results if r.get("latency_ms")]
        if all_lats:
            lines.append(
                f"| **Overall** | **{sum(all_lats) / len(all_lats):.0f}** | **{min(all_lats):.0f}** | **{max(all_lats):.0f}** |"
            )

    # RAGAS
    if ragas_data:
        lines += [
            "",
            "---",
            "",
            "## 3. RAGAS Quality Scores",
            "",
            "| Metric | Score |",
            "|---|---|",
        ]
        for k, v in ragas_data.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.3f} |")

    # Conclusion
    lines += [
        "",
        "---",
        "",
        "## 4. Summary & DOD Compliance",
        "",
        "| Criterion | Target | Actual | Status |",
        "|---|---|---|---|",
        f"| Hit Rate @5 (overall) | ≥ 0.60 | {h_overall['hit_rate']:.3f} | {'✅ PASS' if h_overall['hit_rate'] >= 0.60 else '❌ FAIL'} |",
    ]

    if all_lats:
        avg_lat = sum(all_lats) / len(all_lats)
        lines.append(
            f"| Retrieval Latency | < 1000 ms | {avg_lat:.0f} ms | {'✅ PASS' if avg_lat < 1000 else '❌ FAIL'} |"
        )

    if ragas_data:
        faith = ragas_data.get("faithfulness", 0)
        relev = ragas_data.get("answer_relevancy", 0)
        lines.append(
            f"| Faithfulness | ≥ 0.80 | {faith:.3f} | {'✅ PASS' if faith >= 0.80 else '❌ FAIL'} |"
        )
        lines.append(
            f"| Answer Relevancy | ≥ 0.75 | {relev:.3f} | {'✅ PASS' if relev >= 0.75 else '❌ FAIL'} |"
        )

    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Hybrid-RAG evaluation")
    parser.add_argument("--repo", default="llama-core", help="Repository namespace")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K retrieval")
    parser.add_argument("--hit-at-k", type=int, default=5, help="Hit Rate @K")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS scoring")
    parser.add_argument(
        "--ragas-samples",
        type=int,
        default=0,
        help="Limit RAGAS to the first N cases for smoke testing; 0 evaluates all cases",
    )
    parser.add_argument(
        "--ragas-judge-model",
        default=os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_LLM_MODEL),
    )
    parser.add_argument(
        "--ragas-embedding-model",
        default=os.environ.get("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL),
    )
    parser.add_argument(
        "--skip-vector-baseline", action="store_true", help="Skip vector-only baseline"
    )
    parser.add_argument(
        "--falkordb-host",
        default=os.environ.get("FALKORDB_HOST", "localhost"),
    )
    parser.add_argument(
        "--falkordb-port",
        type=int,
        default=int(os.environ.get("FALKORDB_PORT", "6379")),
    )
    parser.add_argument(
        "--falkordb-graph",
        default=os.environ.get("FALKORDB_GRAPH", "codebase"),
    )
    parser.add_argument(
        "--qdrant-host",
        default=os.environ.get("QDRANT_HOST", "localhost"),
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.environ.get("QDRANT_PORT", "6333")),
    )
    parser.add_argument(
        "--qdrant-collection",
        default=os.environ.get("QDRANT_COLLECTION", "code_chunks"),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Connect to stores ─────────────────────────────────────────
    logger.info("Connecting to FalkorDB at %s:%d ...", args.falkordb_host, args.falkordb_port)
    graph = FalkorDBStore(
        host=args.falkordb_host,
        port=args.falkordb_port,
        graph_name=args.falkordb_graph,
    )

    logger.info("Connecting to Qdrant at %s:%d ...", args.qdrant_host, args.qdrant_port)
    vector = QdrantStore(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
    )

    logger.info("Creating OllamaEmbedder at %s ...", args.ollama_url)
    embedder = OllamaEmbedder(ollama_url=args.ollama_url)

    logger.info("Creating HybridRetriever ...")
    retriever = HybridRetriever(
        graph_store=graph,
        vector_store=vector,
        embedder=embedder,
    )
    index_metadata = validate_index_provenance(graph, vector, args.repo)
    ground_truth = {case.id: compute_ground_truth(graph, case, args.repo) for case in EVAL_CORPUS}
    require_complete_ground_truth(ground_truth)

    # ── 1. Hybrid Hit Rate ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Hybrid-RAG Hit Rate @%d evaluation", args.hit_at_k)
    logger.info("=" * 60)
    hybrid_results = evaluate_hit_rate(
        retriever,
        graph,
        EVAL_CORPUS,
        top_k=args.top_k,
        hit_at_k=args.hit_at_k,
        repository=args.repo,
    )

    # ── 2. Vector-only Baseline ───────────────────────────────────
    vector_results = []
    if not args.skip_vector_baseline:
        logger.info("=" * 60)
        logger.info("PHASE 2: Vector-only Baseline Hit Rate @%d", args.hit_at_k)
        logger.info("=" * 60)
        vector_results = evaluate_vector_only(
            vector,
            graph,
            EVAL_CORPUS,
            top_k=args.top_k,
            hit_at_k=args.hit_at_k,
            ollama_url=args.ollama_url,
            repository=args.repo,
        )
    else:
        # Placeholder
        vector_results = [
            {"id": c.id, "hops": c.hops, "query_type": c.query_type, "hit": False, "gt_count": 0}
            for c in EVAL_CORPUS
        ]

    # ── 3. RAGAS (optional) ───────────────────────────────────────
    ragas_data = None
    if not args.skip_ragas:
        logger.info("=" * 60)
        logger.info("PHASE 3: RAGAS quality evaluation")
        logger.info("=" * 60)
        from hybrid_rag.eval.ragas_runner import RagasRunner

        ragas_cases = EVAL_CORPUS[: args.ragas_samples] if args.ragas_samples else EVAL_CORPUS
        references = {
            case.id: build_reference_answer(case, ground_truth[case.id]) for case in ragas_cases
        }
        runner = RagasRunner(
            retriever=retriever,
            repository=args.repo,
            ollama_url=args.ollama_url,
            judge_model=args.ragas_judge_model,
            embedding_model=args.ragas_embedding_model,
        )
        report = runner.run(
            ragas_cases,
            references=references,
            top_k=args.top_k,
        )
        ragas_data = report.as_dict()

    # ── Save results ──────────────────────────────────────────────
    full_results = {
        "timestamp": timestamp,
        "config": {
            "repo": args.repo,
            "top_k": args.top_k,
            "hit_at_k": args.hit_at_k,
            "dataset": "LlamaIndex core",
            "n_queries": len(EVAL_CORPUS),
            "falkordb_graph": args.falkordb_graph,
            "qdrant_collection": args.qdrant_collection,
            "index_metadata": index_metadata,
        },
        "hybrid_hit_rate": compute_hit_rate(hybrid_results, "Hybrid-RAG"),
        "vector_hit_rate": compute_hit_rate(vector_results, "Vector-only RAG"),
        "hybrid_per_query": hybrid_results,
        "vector_per_query": vector_results,
        "ragas": ragas_data,
    }

    json_path = RESULTS_DIR / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("JSON results saved to %s", json_path)

    md_path = RESULTS_DIR / "evaluation_report.md"
    report_md = generate_report(
        hybrid_results,
        vector_results,
        hybrid_results,
        ragas_data,
        timestamp,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Markdown report saved to %s", md_path)

    # Print summary
    h_summary = compute_hit_rate(hybrid_results, "Hybrid-RAG")
    v_summary = compute_hit_rate(vector_results, "Vector-only")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(
        f"Hybrid-RAG  Hit Rate @{args.hit_at_k}: {h_summary['overall']['hit_rate']:.3f} ({h_summary['overall']['hits']}/{h_summary['overall']['total']})"
    )
    print(
        f"Vector-only Hit Rate @{args.hit_at_k}: {v_summary['overall']['hit_rate']:.3f} ({v_summary['overall']['hits']}/{v_summary['overall']['total']})"
    )
    print(f"\nResults: {json_path}")
    print(f"Report:  {md_path}")


if __name__ == "__main__":
    main()
