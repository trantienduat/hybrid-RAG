"""
Unit tests for the RepoQA evaluation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.eval.corpus import RepoQACase, load_repoqa_json
from hybrid_rag.eval.metrics import RepoQAEvalReport, RepoQAQueryResult
from hybrid_rag.eval.runner import RepoQAEvalRunner


def test_load_repoqa_json(tmp_path: Path):
    # Prepare a mock JSON file
    data = [
        {
            "id": "repoqa-1",
            "question": "A helper function to sum two values.",
            "target_function": "add",
            "file_path": "math_utils.py",
            "notes": "notes context",
        },
        {
            "id": "repoqa-2",
            "question": "A helper function to subtract.",
            "target_function": "sub",
            "file_path": "math_utils.py",
        },
    ]
    json_file = tmp_path / "repoqa.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    cases = load_repoqa_json(json_file)
    assert len(cases) == 2
    assert cases[0].id == "repoqa-1"
    assert cases[0].target_function == "add"
    assert cases[0].file_path == "math_utils.py"
    assert cases[0].notes == "notes context"

    assert cases[1].id == "repoqa-2"
    assert cases[1].target_function == "sub"
    assert cases[1].file_path == "math_utils.py"
    assert cases[1].notes == ""


def test_load_repoqa_json_missing():
    with pytest.raises(FileNotFoundError):
        load_repoqa_json("nonexistent_file_path_12345.json")


def test_repoqa_query_result_metrics():
    # Test rank == 1
    r1 = RepoQAQueryResult(
        query_id="q1",
        question="q",
        target_function="add",
        file_path="math.py",
        rank_hybrid=1,
        rank_vector=0,
    )
    assert r1.hit_at1_hybrid == 1.0
    assert r1.hit_at1_vector == 0.0
    assert r1.hit_at5_hybrid == 1.0
    assert r1.hit_at5_vector == 0.0
    assert r1.mrr_hybrid == 1.0
    assert r1.mrr_vector == 0.0

    # Test rank == 3 vs rank == 6
    r2 = RepoQAQueryResult(
        query_id="q2",
        question="q",
        target_function="sub",
        file_path="math.py",
        rank_hybrid=3,
        rank_vector=6,
    )
    assert r2.hit_at1_hybrid == 0.0
    assert r2.hit_at1_vector == 0.0
    assert r2.hit_at5_hybrid == 1.0
    assert r2.hit_at5_vector == 0.0
    assert abs(r2.mrr_hybrid - (1.0 / 3.0)) < 1e-9
    assert abs(r2.mrr_vector - (1.0 / 6.0)) < 1e-9


def test_repoqa_eval_report_agg():
    r1 = RepoQAQueryResult(
        query_id="q1",
        question="q",
        target_function="add",
        file_path="math.py",
        rank_hybrid=1,
        rank_vector=3,
    )
    r2 = RepoQAQueryResult(
        query_id="q2",
        question="q",
        target_function="sub",
        file_path="math.py",
        rank_hybrid=0,
        rank_vector=6,
    )
    report = RepoQAEvalReport(results=[r1, r2])

    # Hit@1: hybrid = [1, 0] -> 0.5, vector = [0, 0] -> 0.0
    assert report.hit1_hybrid == 0.5
    assert report.hit1_vector == 0.0

    # Hit@5: hybrid = [1, 0] -> 0.5, vector = [1, 0] -> 0.5
    assert report.hit5_hybrid == 0.5
    assert report.hit5_vector == 0.5

    # MRR: hybrid = [1.0, 0.0] -> 0.5, vector = [1/3, 1/6] -> (0.5 / 2) = 0.25
    assert report.mrr_hybrid == 0.5
    assert report.mrr_vector == 0.25


def test_find_target_rank():
    # Setup runner with mocked dependencies
    graph_store = MagicMock()
    vector_store = MagicMock()
    embedder = MagicMock()
    runner = RepoQAEvalRunner(graph_store, vector_store, embedder)

    case = RepoQACase(
        id="q1",
        question="...",
        target_function="add_numbers",
        file_path="math_utils.py",
    )

    # 1. Exact match by name & ending file path
    results = [
        {"name": "sub", "file_path": "math_utils.py"},
        {"name": "add_numbers", "file_path": "other.py"},
        {"name": "add_numbers", "file_path": "src/math_utils.py"},  # Should match rank=3
    ]
    rank = runner._find_target_rank(results, case)
    assert rank == 3

    # 2. Match by node_id ending
    results_node_id = [
        {"node_id": "math_utils.py::sub", "file_path": "math_utils.py"},
        {"node_id": "math_utils.py::add_numbers", "file_path": "math_utils.py"},  # Matches rank=2
    ]
    rank_node = runner._find_target_rank(results_node_id, case)
    assert rank_node == 2

    # 3. Match by base_node_id double colon pattern
    results_base = [
        {
            "base_node_id": "src/math_utils.py::add_numbers::0",
            "file_path": "src/math_utils.py",
        },  # Matches rank=1
    ]
    rank_base = runner._find_target_rank(results_base, case)
    assert rank_base == 1

    # 4. No match
    results_none = [
        {"name": "add", "file_path": "math_utils.py"},
    ]
    assert runner._find_target_rank(results_none, case) == 0


@patch("hybrid_rag.eval.runner.HybridRetriever")
def test_repoqa_runner_run(mock_retriever_class):
    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever

    # Hybrid returns math_utils.py::add_numbers
    # Vector returns other_file.py::other
    mock_retriever.retrieve.side_effect = lambda q, top_k, skip_graph=False, **kwargs: (
        [
            {"name": "other_func", "file_path": "math_utils.py"},
            {"name": "add_numbers", "file_path": "math_utils.py"},
        ]
        if not skip_graph
        else [{"name": "other_func", "file_path": "math_utils.py"}]
    )

    graph_store = MagicMock()
    vector_store = MagicMock()
    embedder = MagicMock()
    runner = RepoQAEvalRunner(graph_store, vector_store, embedder)

    case = RepoQACase(
        id="q1",
        question="q",
        target_function="add_numbers",
        file_path="math_utils.py",
    )

    report = runner.run([case], top_k=5)
    assert len(report.results) == 1
    res = report.results[0]
    assert res.rank_hybrid == 2
    assert res.rank_vector == 0
    assert res.hit_at1_hybrid == 0.0
    assert res.hit_at5_hybrid == 1.0
    assert res.mrr_hybrid == 0.5
    assert res.mrr_vector == 0.0
