"""
Latency benchmark — M4 #32.

Measures p50 / p95 / p99 retrieval latency (ms) across query types.

Usage::

    runner = BenchmarkRunner(retriever)
    report = runner.run(n_runs=5, top_k=20)
    print(report.summary())
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from hybrid_rag.eval.corpus import EVAL_CORPUS, QueryCase

logger = logging.getLogger(__name__)

# Representative queries per type used for benchmarking
_BENCH_QUERIES: dict[str, list[str]] = {
    "structural": [
        "Which classes directly inherit from BaseSynthesizer?",
        "What methods does the BaseRetriever class define?",
        "What modules does llama_index/core/query_engine/retriever_query_engine.py import?",
    ],
    "hybrid": [
        "Explain the data flow when a user submits a query: which classes are instantiated?",
        "Find all classes that implement a retry or fallback mechanism.",
        "Which functions access the file system directly?",
    ],
    "semantic": [
        "Explain how the embedding pipeline works end-to-end.",
        "What is the purpose of the EntityResolver component?",
        "How does the context assembler format results for the LLM?",
    ],
}


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class LatencyStats:
    query_type: str
    n_runs: int
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    all_ms: list[float] = field(default_factory=list, repr=False)


@dataclass
class BenchmarkReport:
    stats: list[LatencyStats] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Latency Benchmark Results", "-" * 50]
        for s in self.stats:
            lines.append(
                f"{s.query_type:12s}  n={s.n_runs:3d}  "
                f"p50={s.p50:6.1f}ms  p95={s.p95:6.1f}ms  "
                f"p99={s.p99:6.1f}ms  mean={s.mean:6.1f}ms"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            s.query_type: {
                "n_runs": s.n_runs,
                "p50_ms": round(s.p50, 2),
                "p95_ms": round(s.p95, 2),
                "p99_ms": round(s.p99, 2),
                "mean_ms": round(s.mean, 2),
                "min_ms": round(s.min, 2),
                "max_ms": round(s.max, 2),
            }
            for s in self.stats
        }


# ── Runner ────────────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """Measure retrieval latency across structural / hybrid / semantic queries."""

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def run(
        self,
        n_runs: int = 5,
        top_k: int = 20,
        warmup_runs: int = 1,
        queries: dict[str, list[str]] | None = None,
    ) -> BenchmarkReport:
        """
        Measure p50/p95/p99 latency for each query type.

        Args:
            n_runs:       Timed runs per query per type.
            top_k:        Retrieval candidates (matches production config).
            warmup_runs:  Untimed warm-up runs to prime caches.
            queries:      Override default benchmark queries.
        """
        query_map = queries or _BENCH_QUERIES
        stats: list[LatencyStats] = []

        for qtype, qs in query_map.items():
            all_ms: list[float] = []
            for q in qs:
                # Warm-up
                for _ in range(warmup_runs):
                    try:
                        self._retriever.retrieve(q, top_k=top_k)
                    except Exception:  # noqa: BLE001
                        pass
                # Timed runs
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    try:
                        self._retriever.retrieve(q, top_k=top_k)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Retrieval failed during benchmark: %s", exc)
                    elapsed = (time.perf_counter() - t0) * 1000
                    all_ms.append(elapsed)

            if all_ms:
                all_ms_sorted = sorted(all_ms)
                n = len(all_ms_sorted)
                stats.append(
                    LatencyStats(
                        query_type=qtype,
                        n_runs=n,
                        p50=_percentile(all_ms_sorted, 50),
                        p95=_percentile(all_ms_sorted, 95),
                        p99=_percentile(all_ms_sorted, 99),
                        mean=statistics.mean(all_ms),
                        min=all_ms_sorted[0],
                        max=all_ms_sorted[-1],
                        all_ms=all_ms,
                    )
                )
                logger.info(
                    "Benchmark %s: n=%d p50=%.1fms p95=%.1fms p99=%.1fms",
                    qtype,
                    n,
                    stats[-1].p50,
                    stats[-1].p95,
                    stats[-1].p99,
                )

        return BenchmarkReport(stats=stats)

    def run_from_corpus(
        self,
        corpus: list[QueryCase] | None = None,
        n_runs: int = 3,
        top_k: int = 20,
    ) -> BenchmarkReport:
        """Run benchmark using the actual eval corpus queries, grouped by query_type."""
        cases = corpus or EVAL_CORPUS
        queries: dict[str, list[str]] = {}
        for c in cases:
            queries.setdefault(c.query_type, []).append(c.question)
        return self.run(n_runs=n_runs, top_k=top_k, queries=queries)


# ── Utility ───────────────────────────────────────────────────────────────────


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac
