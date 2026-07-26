"""Unit tests for the standalone full-evaluation report helpers."""

from scripts.run_full_evaluation import compute_hit_rate


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
