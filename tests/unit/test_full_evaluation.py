"""Unit tests for the standalone full-evaluation report helpers."""

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
from scripts.run_full_evaluation import compute_hit_rate


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
