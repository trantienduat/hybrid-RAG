"""
Reciprocal Rank Fusion (RRF) — merge multiple ranked lists into one.

Formula::

    score(d) = Σ_{r ∈ R}  1 / (k + rank_r(d))

where k = 60 by default prevents high variance from top-rank positions.

Results are sorted by rrf_score descending; ties broken by first-seen order.

Reference: Cormack, Clarke & Buettcher — "Reciprocal Rank Fusion outperforms
Condorcet and individual rank learning methods", SIGIR 2009.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_K = 60


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    id_key: str = "base_node_id",
    k: int = _DEFAULT_K,
    weights: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    Merge *ranked_lists* using Reciprocal Rank Fusion.

    Each list element must have a non-empty string at *id_key*.
    Items missing *id_key* or with an empty value are silently skipped.

    *weights* — optional per-list multipliers (same length as ranked_lists).
    Defaults to 1.0 for every list.  Use e.g. ``weights=(3.0, 1.0)`` to give
    the graph list 3× more influence than the vector list for structural queries.

    When the same document appears in multiple lists, the result dict is
    merged with the following rules:
    - ``text``: prefer the first non-empty value seen.
    - ``source``: set to "hybrid" when two different sources contribute.
    - ``rel``: prefer the first non-empty value seen.
    - ``rrf_score``: sum of per-list contributions.

    Returns a new list sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        w = weights[list_idx] if weights and list_idx < len(weights) else 1.0
        for rank, item in enumerate(ranked, start=1):
            doc_id = item.get(id_key, "")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            if doc_id not in merged:
                merged[doc_id] = dict(item)
            else:
                existing = merged[doc_id]
                # Prefer vector text over empty graph text
                if not existing.get("text") and item.get("text"):
                    existing["text"] = item["text"]
                # Merge source label
                ex_src = existing.get("source", "")
                new_src = item.get("source", "")
                if ex_src and new_src and ex_src != new_src:
                    existing["source"] = "hybrid"
                elif not ex_src and new_src:
                    existing["source"] = new_src
                # Prefer first non-empty rel
                if not existing.get("rel") and item.get("rel"):
                    existing["rel"] = item["rel"]
                # Prefer first non-empty name
                if not existing.get("name") and item.get("name"):
                    existing["name"] = item["name"]
                # Prefer first non-empty label
                if not existing.get("label") and item.get("label"):
                    existing["label"] = item["label"]
                # Prefer first non-empty file_path
                if not existing.get("file_path") and item.get("file_path"):
                    existing["file_path"] = item["file_path"]
                # Prefer first non-empty repository
                if not existing.get("repository") and item.get("repository"):
                    existing["repository"] = item["repository"]

    for doc_id, item in merged.items():
        item["rrf_score"] = scores[doc_id]

    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
