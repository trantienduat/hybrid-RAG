#!/usr/bin/env python3
"""Compute paired case-level confidence intervals for answer-quality results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results/answer_quality_benchmark_schema3.json"
DEFAULT_OUTPUT = ROOT / "results/answer_quality_benchmark_statistics.json"

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "prompt_tokens",
    "completion_tokens",
)
MODES = ("vector", "hybrid")


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _case_means(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = str(record.get("case_id", "")).strip()
        mode = str(record.get("mode", "")).strip()
        if not case_id or mode not in MODES:
            raise ValueError(f"Invalid case or mode: {case_id!r}/{mode!r}")
        grouped[(case_id, mode)].append(record)

    case_ids = sorted({case_id for case_id, _ in grouped})
    if not case_ids:
        raise ValueError("No benchmark records found")

    means: dict[str, dict[str, dict[str, float]]] = {}
    repeat_counts: set[int] = set()
    for case_id in case_ids:
        means[case_id] = {}
        for mode in MODES:
            mode_records = grouped.get((case_id, mode), [])
            if not mode_records:
                raise ValueError(f"Missing {mode} records for {case_id}")
            repeat_counts.add(len(mode_records))
            means[case_id][mode] = {}
            for metric in METRICS:
                values = [record.get(metric) for record in mode_records]
                if any(
                    not isinstance(value, int | float) or not math.isfinite(float(value))
                    for value in values
                ):
                    raise ValueError(f"Invalid {metric} value for {case_id}/{mode}")
                means[case_id][mode][metric] = fmean(float(value) for value in values)

    if len(repeat_counts) != 1:
        raise ValueError(f"Unbalanced repeat counts: {sorted(repeat_counts)}")
    return case_ids, {"means": means, "repeats_per_case_mode": repeat_counts.pop()}


def analyze(
    payload: dict[str, Any], *, resamples: int, seed: int, confidence_level: float
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Input must contain a records list")
    case_ids, grouped = _case_means(records)
    means = grouped["means"]
    rng = random.Random(seed)
    alpha = (1 - confidence_level) / 2

    metric_results: dict[str, Any] = {}
    for metric in METRICS:
        vector_values = [means[case_id]["vector"][metric] for case_id in case_ids]
        hybrid_values = [means[case_id]["hybrid"][metric] for case_id in case_ids]
        vector_bootstrap: list[float] = []
        hybrid_bootstrap: list[float] = []
        delta_bootstrap: list[float] = []

        for _ in range(resamples):
            sampled_indexes = [rng.randrange(len(case_ids)) for _ in case_ids]
            vector_mean = fmean(vector_values[index] for index in sampled_indexes)
            hybrid_mean = fmean(hybrid_values[index] for index in sampled_indexes)
            vector_bootstrap.append(vector_mean)
            hybrid_bootstrap.append(hybrid_mean)
            delta_bootstrap.append(hybrid_mean - vector_mean)

        vector_mean = fmean(vector_values)
        hybrid_mean = fmean(hybrid_values)
        metric_results[metric] = {
            "vector_mean": vector_mean,
            "vector_ci": [
                _percentile(vector_bootstrap, alpha),
                _percentile(vector_bootstrap, 1 - alpha),
            ],
            "hybrid_mean": hybrid_mean,
            "hybrid_ci": [
                _percentile(hybrid_bootstrap, alpha),
                _percentile(hybrid_bootstrap, 1 - alpha),
            ],
            "delta_hybrid_minus_vector": hybrid_mean - vector_mean,
            "delta_ci": [
                _percentile(delta_bootstrap, alpha),
                _percentile(delta_bootstrap, 1 - alpha),
            ],
        }

    return {
        "schema_version": 1,
        "method": "paired nonparametric bootstrap over case-level means",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "cases": len(case_ids),
        "repeats_per_case_mode": grouped["repeats_per_case_mode"],
        "records": len(records),
        "metrics": metric_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired case-level bootstrap confidence intervals."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    payload = json.loads(source_bytes)
    result = analyze(
        payload,
        resamples=args.resamples,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )
    result["source"] = {
        "path": str(args.input.relative_to(ROOT)),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "generated_at": payload.get("generated_at"),
        "evidence_grade": payload.get("evidence_grade"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
