"""Ground-truth helpers shared by evaluation entry points."""

from __future__ import annotations

import logging

from hybrid_rag.eval.corpus import QueryCase
from hybrid_rag.ports.graph_store import GraphStore

logger = logging.getLogger(__name__)


def compute_ground_truth(
    graph: GraphStore,
    case: QueryCase,
    repository: str,
) -> set[str]:
    """Execute a case's Cypher oracle and return normalized entity names."""
    try:
        raw = graph.query(case.ground_truth_cypher, {"repository": repository})
        rows = getattr(raw, "result_set", []) or []
        return {
            value.strip().lower()
            for row in rows
            for value in [row[case.gt_col_index] if isinstance(row, (list, tuple)) else row]
            if isinstance(value, str) and value.strip()
        }
    except Exception as exc:
        logger.warning("Ground-truth query failed for %s: %s", case.id, exc)
        return set()


def build_reference_answer(case: QueryCase, expected_names: set[str]) -> str:
    """Build a deterministic, human-readable reference from a graph oracle."""
    if not expected_names:
        raise ValueError(f"Cannot build reference answer for {case.id} without ground truth")
    names = ", ".join(sorted(expected_names))
    return f"For the question '{case.question}', the relevant code entities are: {names}."
