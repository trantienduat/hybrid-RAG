"""Validation gates for reproducible retrieval evaluations."""

from __future__ import annotations

from typing import Any

from hybrid_rag.constants import INDEX_SCHEMA_VERSION
from hybrid_rag.ports.graph_store import GraphStore
from hybrid_rag.ports.vector_store import VectorStore


class EvaluationPreflightError(RuntimeError):
    """Raised when an evaluation would produce an untrustworthy score."""


def _stable_values(values: set[Any]) -> list[str]:
    return sorted(repr(value) for value in values)


def validate_index_provenance(
    graph_store: GraphStore,
    vector_store: VectorStore,
    repository: str,
    *,
    expected_embedding_model: str | None = None,
) -> dict[str, Any]:
    """Require one matching provenance identity across graph and vectors."""
    metadata = graph_store.get_repository_metadata(repository) or {}
    required = ("source_identity", "index_run_id", "index_schema_version", "embedding_model")
    missing = [field for field in required if metadata.get(field) in (None, "")]
    errors = [f"graph metadata missing {field}" for field in missing]

    if metadata.get("working_tree_dirty"):
        errors.append("indexed source had uncommitted changes")
    if metadata.get("index_schema_version") != INDEX_SCHEMA_VERSION:
        errors.append(
            f"graph index_schema_version {metadata.get('index_schema_version')!r} "
            f"does not match current schema {INDEX_SCHEMA_VERSION!r}"
        )
    if expected_embedding_model and metadata.get("embedding_model") != expected_embedding_model:
        errors.append(
            f"graph embedding_model {metadata.get('embedding_model')!r} does not match "
            f"requested model {expected_embedding_model!r}"
        )

    vector_metadata = vector_store.get_repository_metadata(repository)
    for field in (
        "source_identity",
        "index_run_id",
        "index_schema_version",
        "embedding_model",
    ):
        values = vector_metadata.get(field, set())
        expected = metadata.get(field)
        if values != ({expected} if expected else set()):
            errors.append(
                f"vector {field} values {_stable_values(values)!r} "
                f"do not match graph value {expected!r}"
            )

    indexed_commit = metadata.get("last_indexed_commit")
    vector_commits = vector_metadata.get("indexed_commit", set())
    if indexed_commit and vector_commits != {indexed_commit}:
        errors.append(
            f"vector indexed_commit values {_stable_values(vector_commits)!r} "
            f"do not match graph value {indexed_commit!r}"
        )

    if errors:
        detail = "; ".join(errors)
        raise EvaluationPreflightError(
            f"Index provenance check failed for {repository}: {detail}. Rebuild the index."
        )
    return metadata


def require_complete_ground_truth(ground_truth: dict[str, set[str]]) -> None:
    """Reject partial corpora instead of silently improving the denominator."""
    empty = [case_id for case_id, names in ground_truth.items() if not names]
    if empty:
        raise EvaluationPreflightError(
            "Ground truth is empty for "
            f"{', '.join(empty)}. Repair the corpus or index before evaluating."
        )


def require_same_index_run(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Reject measurements collected across different index snapshots."""
    if before.get("index_run_id") != after.get("index_run_id"):
        raise EvaluationPreflightError("Index changed during measurement; discard it and retry")
