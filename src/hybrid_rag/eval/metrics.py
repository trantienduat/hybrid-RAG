"""
Evaluation metrics — Hit Rate @K and Mean Reciprocal Rank (MRR).

All functions are pure (no I/O), operate on pre-computed string lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def hit_rate_at_k(retrieved_names: list[str], relevant: set[str], k: int = 5) -> float:
    """
    Return 1.0 if any of the top-k retrieved names is in *relevant*, else 0.0.

    Names are compared case-insensitively after stripping whitespace.
    Returns 0.0 when *relevant* is empty (undefined — no ground truth).
    """
    if not relevant:
        return 0.0
    relevant_lower = {r.strip().lower() for r in relevant}
    for name in retrieved_names[:k]:
        if name.strip().lower() in relevant_lower:
            return 1.0
    return 0.0


def mrr_score(retrieved_names: list[str], relevant: set[str]) -> float:
    """
    Reciprocal rank of the first relevant result in the list.

    Returns 0.0 when *relevant* is empty or no relevant result is found.
    """
    if not relevant:
        return 0.0
    relevant_lower = {r.strip().lower() for r in relevant}
    for rank, name in enumerate(retrieved_names, start=1):
        if name.strip().lower() in relevant_lower:
            return 1.0 / rank
    return 0.0


@dataclass
class QueryResult:
    """Per-query evaluation outcome for one retrieval mode."""
    query_id: str
    question: str
    hops: int
    query_type: str
    ground_truth: set[str]     # node names from Cypher
    retrieved_hybrid: list[str]
    retrieved_vector: list[str]
    hit_at5_hybrid: float = 0.0
    hit_at5_vector: float = 0.0
    mrr_hybrid: float = 0.0
    mrr_vector: float = 0.0
    delta_hit: float = 0.0     # hybrid - vector

    def __post_init__(self) -> None:
        self.hit_at5_hybrid = hit_rate_at_k(self.retrieved_hybrid, self.ground_truth, k=5)
        self.hit_at5_vector = hit_rate_at_k(self.retrieved_vector, self.ground_truth, k=5)
        self.mrr_hybrid     = mrr_score(self.retrieved_hybrid, self.ground_truth)
        self.mrr_vector     = mrr_score(self.retrieved_vector, self.ground_truth)
        self.delta_hit      = self.hit_at5_hybrid - self.hit_at5_vector


@dataclass
class EvalReport:
    """Aggregated evaluation report across all queries."""
    results: list[QueryResult] = field(default_factory=list)

    # ── aggregate helpers ─────────────────────────────────────────────────────

    def _avg(self, attr: str, subset: list[QueryResult] | None = None) -> float:
        rows = subset if subset is not None else self.results
        vals = [getattr(r, attr) for r in rows if r.ground_truth]
        return sum(vals) / len(vals) if vals else 0.0

    def _filter_hops(self, hops: int | None, max_hops: int | None = None) -> list[QueryResult]:
        out = []
        for r in self.results:
            if hops is not None and r.hops != hops:
                continue
            if max_hops is not None and r.hops > max_hops:
                continue
            out.append(r)
        return out

    # ── summary properties ────────────────────────────────────────────────────

    @property
    def hit5_hybrid_all(self) -> float:
        return self._avg("hit_at5_hybrid")

    @property
    def hit5_vector_all(self) -> float:
        return self._avg("hit_at5_vector")

    @property
    def mrr_hybrid_all(self) -> float:
        return self._avg("mrr_hybrid")

    @property
    def mrr_vector_all(self) -> float:
        return self._avg("mrr_vector")

    @property
    def delta_hit_2_3hop(self) -> float:
        """ΔHitRate hybrid-vector for 2- and 3-hop queries (thesis target: ≥+0.20)."""
        rows = [r for r in self.results if r.hops >= 2 and r.ground_truth]
        if not rows:
            return 0.0
        return sum(r.delta_hit for r in rows) / len(rows)

    @property
    def hit5_hybrid_by_hops(self) -> dict[int, float]:
        return {h: self._avg("hit_at5_hybrid", self._filter_hops(h)) for h in (1, 2, 3)}

    @property
    def hit5_vector_by_hops(self) -> dict[int, float]:
        return {h: self._avg("hit_at5_vector", self._filter_hops(h)) for h in (1, 2, 3)}
