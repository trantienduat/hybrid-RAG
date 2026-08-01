"""Unit tests for the standalone full-evaluation report helpers."""

from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.eval.corpus import (
    AMBIGUOUS,
    EVAL_CORPUS,
    HYBRID,
    ONE_HOP,
    THREE_HOP,
    TWO_HOP,
)
from hybrid_rag.eval.ground_truth import build_reference_answer
from scripts.run_full_evaluation import (
    compute_hit_rate,
    evaluate_hit_rate,
    evaluate_vector_only,
    generate_report,
)


def test_diagnostic_corpus_has_50_unique_partitioned_cases():
    partitions = ONE_HOP + TWO_HOP + THREE_HOP + HYBRID + AMBIGUOUS

    assert [len(group) for group in (ONE_HOP, TWO_HOP, THREE_HOP, HYBRID, AMBIGUOUS)] == [
        10,
        15,
        10,
        10,
        5,
    ]
    assert len(EVAL_CORPUS) == len(partitions) == 50
    assert len({case.id for case in EVAL_CORPUS}) == 50
    assert {case.id for case in EVAL_CORPUS} == {case.id for case in partitions}
    assert all("$repository" in case.ground_truth_cypher for case in EVAL_CORPUS)


def test_compute_hit_rate_excludes_empty_ground_truth():
    summary = compute_hit_rate(
        [
            {"hops": 1, "hit": True, "gt_count": 2},
            {"hops": 1, "hit": False, "gt_count": 0},
            {"hops": 2, "hit": False, "gt_count": 3},
        ],
        label="Hybrid-RAG",
    )

    assert summary["overall"] == {"hit_rate": 0.5, "hits": 1, "total": 2}
    assert summary["skipped_empty_ground_truth"] == 1
    assert summary["1-hop"]["total"] == 1


def test_build_reference_answer_is_deterministic():
    case = EVAL_CORPUS[0]

    reference = build_reference_answer(case, {"beta", "alpha"})

    assert reference == (
        f"For the question '{case.question}', the relevant code entities are: alpha, beta."
    )


def test_build_reference_answer_rejects_empty_ground_truth():
    with pytest.raises(ValueError, match="without ground truth"):
        build_reference_answer(EVAL_CORPUS[0], set())


@patch("scripts.run_full_evaluation.compute_ground_truth", return_value={"target"})
def test_full_evaluation_aborts_on_hybrid_retrieval_failure(_mock_ground_truth):
    retriever = MagicMock()
    retriever.retrieve.side_effect = RuntimeError("database unavailable")

    with pytest.raises(RuntimeError, match="Hybrid retrieval failed"):
        evaluate_hit_rate(
            retriever,
            MagicMock(),
            [EVAL_CORPUS[0]],
            repository="repo",
        )


@patch("scripts.run_full_evaluation.compute_ground_truth", return_value={"target"})
def test_full_evaluation_aborts_on_vector_embedding_failure(_mock_ground_truth):
    embedder = MagicMock()
    embedder.embed_query.side_effect = RuntimeError("embedding unavailable")

    with pytest.raises(RuntimeError, match="Vector-only retrieval failed"):
        evaluate_vector_only(
            MagicMock(),
            MagicMock(),
            embedder,
            [EVAL_CORPUS[0]],
            repository="repo",
        )


def test_report_records_measurement_configuration_and_provenance():
    result = {
        "id": "Q1",
        "hops": 1,
        "query_type": "local",
        "hit": True,
        "gt_count": 1,
        "latency_ms": 12.5,
    }

    report = generate_report(
        [result],
        [result],
        [result],
        None,
        "2026-08-01T00:00:00Z",
        "test-platform",
        top_k=20,
        hit_at_k=7,
        rrf_k=60,
        rrf_structural_weight=1.5,
        index_metadata={
            "index_run_id": "run-1",
            "index_schema_version": 3,
            "embedding_model": "nomic-embed-text",
        },
    )

    assert "Hit Rate @7 Comparison" in report
    assert "top_k=20, hit_at_k=7, rrf_k=60, structural_weight=1.5" in report
    assert "`run-1` (schema 3, embedding `nomic-embed-text`)" in report
    assert "test-platform" in report
