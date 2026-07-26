"""Smoke tests for the repeatable API concurrency benchmark."""

from __future__ import annotations

import pytest

from scripts.benchmark_api_concurrency import run_benchmark


@pytest.mark.asyncio
async def test_api_concurrency_benchmark_reports_comparable_metrics():
    result = await run_benchmark(requests=2, delay_ms=1, trials=1)

    assert result["config"]["requests"] == 2
    assert result["config"]["trials"] == 1
    assert len(result["before_direct"]["batch_ms_trials"]) == 1
    assert len(result["after_offloaded"]["batch_ms_trials"]) == 1
    assert result["before_direct"]["batch_ms"] > 0
    assert result["after_offloaded"]["batch_ms"] > 0
    assert result["improvement"]["batch_speedup_x"] > 0
