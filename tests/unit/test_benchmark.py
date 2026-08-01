"""Latency benchmark validity tests."""

from unittest.mock import MagicMock

import pytest

from hybrid_rag.eval.benchmark import BenchmarkRunner


def test_benchmark_scopes_every_retrieval_to_repository():
    retriever = MagicMock()
    runner = BenchmarkRunner(retriever, repository="repo-one")

    report = runner.run(
        n_runs=2,
        warmup_runs=1,
        queries={"semantic": ["How does it work?"]},
    )

    assert report.stats[0].n_runs == 2
    assert retriever.retrieve.call_count == 3
    assert all(
        call.kwargs["repository"] == "repo-one" for call in retriever.retrieve.call_args_list
    )


def test_benchmark_fails_instead_of_timing_failed_retrievals():
    retriever = MagicMock()
    retriever.retrieve.side_effect = RuntimeError("database unavailable")
    runner = BenchmarkRunner(retriever, repository="repo-one")

    with pytest.raises(RuntimeError, match="timed retrieval failed"):
        runner.run(
            n_runs=1,
            warmup_runs=0,
            queries={"semantic": ["How does it work?"]},
        )


def test_benchmark_requires_repository():
    with pytest.raises(ValueError, match="repository is required"):
        BenchmarkRunner(MagicMock(), repository=" ")
