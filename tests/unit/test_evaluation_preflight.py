"""Tests for evaluation validity gates."""

from unittest.mock import MagicMock

import pytest

from hybrid_rag.eval.preflight import (
    EvaluationPreflightError,
    require_complete_ground_truth,
    require_same_index_run,
    validate_index_provenance,
)
from hybrid_rag.ingestion.pipeline import INDEX_SCHEMA_VERSION


def _vector_metadata(index_run_id: str = "run-1") -> dict[str, set[object]]:
    return {
        "source_identity": {"git:repo@abc"},
        "index_run_id": {index_run_id},
        "indexed_commit": {"abc"},
        "index_schema_version": {INDEX_SCHEMA_VERSION},
        "embedding_model": {"nomic-embed-text"},
    }


def test_rejects_empty_ground_truth():
    with pytest.raises(EvaluationPreflightError, match="Q3, Q11"):
        require_complete_ground_truth({"Q1": {"retrieve"}, "Q3": set(), "Q11": set()})


def test_rejects_index_change_during_measurement():
    with pytest.raises(EvaluationPreflightError, match="Index changed during measurement"):
        require_same_index_run({"index_run_id": "run-1"}, {"index_run_id": "run-2"})


def test_accepts_unchanged_index_during_measurement():
    require_same_index_run({"index_run_id": "run-1"}, {"index_run_id": "run-1"})


def test_accepts_matching_graph_and_vector_provenance():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-1",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": "nomic-embed-text",
        "last_indexed_commit": "abc",
        "working_tree_dirty": False,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = _vector_metadata()

    metadata = validate_index_provenance(
        graph, vector, "repo", expected_embedding_model="nomic-embed-text"
    )

    assert metadata["index_run_id"] == "run-1"


def test_rejects_mixed_vector_provenance():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-2",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": "nomic-embed-text",
        "last_indexed_commit": "abc",
        "working_tree_dirty": False,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = _vector_metadata()
    vector.get_repository_metadata.return_value["index_run_id"] = {"run-1", "run-2"}

    with pytest.raises(EvaluationPreflightError, match="vector index_run_id"):
        validate_index_provenance(graph, vector, "repo")


def test_rejects_dirty_source_index():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-1",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": "nomic-embed-text",
        "working_tree_dirty": True,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = _vector_metadata()
    vector.get_repository_metadata.return_value["indexed_commit"] = set()

    with pytest.raises(EvaluationPreflightError, match="uncommitted changes"):
        validate_index_provenance(graph, vector, "repo")


def test_rejects_requested_embedding_model_mismatch():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-1",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": "nomic-embed-text",
        "last_indexed_commit": "abc",
        "working_tree_dirty": False,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = _vector_metadata()

    with pytest.raises(EvaluationPreflightError, match="requested model 'other-model'"):
        validate_index_provenance(graph, vector, "repo", expected_embedding_model="other-model")


def test_rejects_vector_embedding_metadata_mismatch():
    graph = MagicMock()
    graph.get_repository_metadata.return_value = {
        "source_identity": "git:repo@abc",
        "index_run_id": "run-1",
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": "nomic-embed-text",
        "last_indexed_commit": "abc",
        "working_tree_dirty": False,
    }
    vector = MagicMock()
    vector.get_repository_metadata.return_value = _vector_metadata()
    vector.get_repository_metadata.return_value["embedding_model"] = {
        "nomic-embed-text",
        "other-model",
    }

    with pytest.raises(EvaluationPreflightError, match="vector embedding_model"):
        validate_index_provenance(graph, vector, "repo")
