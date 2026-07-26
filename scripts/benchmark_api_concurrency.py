#!/usr/bin/env python3
"""Measure event-loop blocking before and after retrieval thread offloading."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from collections.abc import Awaitable, Callable
from typing import Any

from hybrid_rag.api.main import _retrieve_context
from hybrid_rag.api.schemas import QueryRequest
from hybrid_rag.retrieval.context_assembler import RetrievalContext

RetrievalCall = Callable[[Any, QueryRequest], Awaitable[RetrievalContext]]


class BlockingRetriever:
    """Controlled stand-in for blocking graph/vector retrieval."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    def retrieve_with_context(self, *_args: Any, **_kwargs: Any) -> RetrievalContext:
        time.sleep(self._delay_seconds)
        return RetrievalContext(text="")


async def _direct_retrieval(
    retriever: BlockingRetriever,
    request: QueryRequest,
) -> RetrievalContext:
    """Reproduce the previous event-loop-blocking API behavior."""
    return retriever.retrieve_with_context(request.question)


def _nearest_rank(values: list[float], percentile: float) -> float:
    return sorted(values)[max(0, int(percentile * len(values)) - 1)]


async def _measure_trial(
    call: RetrievalCall,
    *,
    requests: int,
    delay_seconds: float,
    heartbeat_seconds: float,
) -> dict[str, float]:
    retriever = BlockingRetriever(delay_seconds)
    request = QueryRequest(question="API concurrency benchmark")
    started = time.perf_counter()
    completions: list[float] = []
    heartbeat_times = [started]
    running = True

    async def heartbeat() -> None:
        while running:
            await asyncio.sleep(heartbeat_seconds)
            heartbeat_times.append(time.perf_counter())

    async def run_request() -> None:
        await call(retriever, request)
        completions.append((time.perf_counter() - started) * 1000)

    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.gather(*(run_request() for _ in range(requests)))
    running = False
    await heartbeat_task

    batch_ms = (time.perf_counter() - started) * 1000
    heartbeat_gaps = [
        (current - previous) * 1000
        for previous, current in zip(heartbeat_times, heartbeat_times[1:])
    ]
    return {
        "batch_ms": batch_ms,
        "throughput_rps": requests / (batch_ms / 1000),
        "completion_p50_ms": statistics.median(completions),
        "completion_p95_ms": _nearest_rank(completions, 0.95),
        "max_event_loop_stall_ms": max(heartbeat_gaps),
        "heartbeat_turns": float(len(heartbeat_times) - 1),
    }


def _mean_metrics(trials: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        metric: round(statistics.mean(trial[metric] for trial in trials), 3) for metric in trials[0]
    }
    summary["batch_ms_trials"] = [round(trial["batch_ms"], 3) for trial in trials]
    return summary


async def run_benchmark(
    *,
    requests: int,
    delay_ms: float,
    trials: int,
    heartbeat_ms: float = 1.0,
) -> dict[str, Any]:
    """Run alternating direct and offloaded trials and return JSON-ready metrics."""
    direct_trials = []
    offloaded_trials = []
    for _ in range(trials):
        direct_trials.append(
            await _measure_trial(
                _direct_retrieval,
                requests=requests,
                delay_seconds=delay_ms / 1000,
                heartbeat_seconds=heartbeat_ms / 1000,
            )
        )
        offloaded_trials.append(
            await _measure_trial(
                _retrieve_context,
                requests=requests,
                delay_seconds=delay_ms / 1000,
                heartbeat_seconds=heartbeat_ms / 1000,
            )
        )

    before = _mean_metrics(direct_trials)
    after = _mean_metrics(offloaded_trials)
    return {
        "config": {
            "requests": requests,
            "blocking_delay_ms": delay_ms,
            "trials": trials,
            "heartbeat_ms": heartbeat_ms,
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "before_direct": before,
        "after_offloaded": after,
        "improvement": {
            "batch_speedup_x": round(before["batch_ms"] / after["batch_ms"], 3),
            "throughput_gain_x": round(after["throughput_rps"] / before["throughput_rps"], 3),
            "event_loop_stall_reduction_pct": round(
                (1 - after["max_event_loop_stall_ms"] / before["max_event_loop_stall_ms"]) * 100,
                3,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--heartbeat-ms", type=float, default=1.0)
    args = parser.parse_args()
    if args.requests < 1 or args.trials < 1 or args.delay_ms <= 0 or args.heartbeat_ms <= 0:
        parser.error("requests and trials must be positive; delays must be greater than zero")

    result = asyncio.run(
        run_benchmark(
            requests=args.requests,
            delay_ms=args.delay_ms,
            trials=args.trials,
            heartbeat_ms=args.heartbeat_ms,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
