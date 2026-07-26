"""Tests for evaluation validity gates."""

from unittest.mock import MagicMock

import pytest

from hybrid_rag.eval.preflight import (
    EvaluationPreflightError,
    require_complete_ground_truth,
    validate_index_provenance,
)


def test_rejects_empty_ground_truth():
    with pytest.raises(EvaluationPreflightError, match="Q3, Q11"):
        require_complete_ground_truth({"Q1": {"retrieve"}, "Q3": set(), "Q11": set()})


def test_accepts_matching_graph_and_vector_provenance():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-1",
        "index_schema_version": 2,
        "embedding_model": "nomic-embed-text",
        "last_indexed_commit": "abc",
        "working_tree_dirty": False,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = {
        "source_identity": {"git:repo@abc"},
        "index_run_id": {"run-1"},
        "indexed_commit": {"abc"},
    }

    metadata = validate_index_provenance(graph, vector, "repo")

    assert metadata["index_run_id"] == "run-1"


def test_rejects_mixed_vector_provenance():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-2",
        "index_schema_version": 2,
        "embedding_model": "nomic-embed-text",
        "last_indexed_commit": "abc",
        "working_tree_dirty": False,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = {
        "source_identity": {"git:repo@abc"},
        "index_run_id": {"run-1", "run-2"},
        "indexed_commit": {"abc"},
    }

    with pytest.raises(EvaluationPreflightError, match="vector index_run_id"):
        validate_index_provenance(graph, vector, "repo")


def test_rejects_dirty_source_index():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-1",
        "index_schema_version": 2,
        "embedding_model": "nomic-embed-text",
        "working_tree_dirty": True,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = {
        "source_identity": {"git:repo@abc"},
        "index_run_id": {"run-1"},
        "indexed_commit": set(),
    }

    with pytest.raises(EvaluationPreflightError, match="uncommitted changes"):
        validate_index_provenance(graph, vector, "repo")
