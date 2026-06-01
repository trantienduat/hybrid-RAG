"""
Evaluation runner — executes both hybrid and vector-only retrieval for each
QueryCase in the corpus and assembles a full EvalReport.

Usage::

    runner = EvalRunner(graph_store, vector_store, embedder)
    report = runner.run(EVAL_CORPUS, top_k=10)
"""
from __future__ import annotations

import logging
from typing import Any

from hybrid_rag.eval.corpus import QueryCase, RepoQACase
from hybrid_rag.eval.metrics import EvalReport, QueryResult, RepoQAEvalReport, RepoQAQueryResult
from hybrid_rag.ports.embedder import BaseEmbedder
from hybrid_rag.ports.graph_store import GraphStore
from hybrid_rag.ports.vector_store import VectorStore
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def _extract_names(results: list[dict[str, Any]]) -> list[str]:
    """Extract a ranked list of node names from retrieval results."""
    names: list[str] = []
    for r in results:
        name = r.get("name", "") or r.get("node_id", "")
        # Strip chunk-index suffix (e.g. "::0") from node_id used as name
        if "::" in name and name.rsplit("::", 1)[-1].isdigit():
            name = name.rsplit("::", 1)[0]
        
        # If FQN dot-notation is used, get the last part (simple name)
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        elif "::" in name:
            name = name.rsplit("::", 1)[-1]
        if name:
            names.append(name)
    return names


class EvalRunner:
    """
    Run baseline evaluation: hybrid vs vector-only on a list of QueryCases.

    *rrf_k* and *rrf_structural_weight* are exposed so the caller (or #26 tuner)
    can sweep parameter values.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        rrf_k: int = 60,
        rrf_structural_weight: float = 3.0,
        rrf_hybrid_weight: float = 1.5,
    ) -> None:
        self._graph_store = graph_store
        self._vector_retriever = VectorRetriever(vector_store, embedder)
        self._hybrid_retriever = HybridRetriever(
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
            rrf_k=rrf_k,
            rrf_structural_weight=rrf_structural_weight,
            rrf_hybrid_weight=rrf_hybrid_weight,
        )

    def _ground_truth(self, case: QueryCase) -> set[str]:
        """Execute the case's Cypher against FalkorDB; return set of expected names."""
        try:
            res = self._graph_store.query(case.ground_truth_cypher)
            names: set[str] = set()
            for row in (res.result_set or []):
                val = row[case.gt_col_index] if row else None
                if val and isinstance(val, str) and val.strip():
                    names.add(val.strip())
            return names
        except Exception as exc:
            logger.warning("Ground truth Cypher failed for %s: %s", case.id, exc)
            return set()

    def run(
        self,
        cases: list[QueryCase],
        top_k: int = 10,
    ) -> EvalReport:
        """
        Run evaluation across *cases*.

        For each case:
        1. Compute ground truth via Cypher.
        2. Run hybrid retrieval (graph + vector RRF).
        3. Run vector-only retrieval (no graph).
        4. Build QueryResult with hit@5 and MRR for both modes.
        """
        report = EvalReport()
        for case in cases:
            logger.info("Evaluating %s: %s", case.id, case.question[:60])

            gt = self._ground_truth(case)
            logger.debug("%s ground truth (%d items): %s", case.id, len(gt), list(gt)[:5])

            # Hybrid mode
            hybrid_results = self._hybrid_retriever.retrieve(case.question, top_k=top_k)
            hybrid_names = _extract_names(hybrid_results)

            # Vector-only mode (skip graph retrieval)
            vector_results = self._hybrid_retriever.retrieve(
                case.question, top_k=top_k, skip_graph=True
            )
            vector_names = _extract_names(vector_results)

            qr = QueryResult(
                query_id=case.id,
                question=case.question,
                hops=case.hops,
                query_type=case.query_type,
                ground_truth=gt,
                retrieved_hybrid=hybrid_names,
                retrieved_vector=vector_names,
            )
            report.results.append(qr)
            logger.info(
                "%s  hit@5 hybrid=%.1f vector=%.1f  Δ=%+.1f  gt=%d",
                case.id,
                qr.hit_at5_hybrid,
                qr.hit_at5_vector,
                qr.delta_hit,
                len(gt),
            )

        return report


class RepoQAEvalRunner:
    """
    Run evaluation on RepoQA: hybrid vs vector-only on a list of RepoQACases.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        rrf_k: int = 60,
        rrf_structural_weight: float = 3.0,
        rrf_hybrid_weight: float = 1.5,
    ) -> None:
        self._graph_store = graph_store
        self._hybrid_retriever = HybridRetriever(
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
            rrf_k=rrf_k,
            rrf_structural_weight=rrf_structural_weight,
            rrf_hybrid_weight=rrf_hybrid_weight,
        )

    def _find_target_rank(self, results: list[dict[str, Any]], case: RepoQACase) -> int:
        """Find the 1-based rank of the first chunk/node matching target function in results."""
        target_func = case.target_function
        target_file = case.file_path

        for rank, item in enumerate(results, start=1):
            # 1. Match file path
            item_file = item.get("file_path", "") or ""
            file_match = False
            if item_file:
                file_match = item_file.lower().endswith(target_file.lower())

            # 2. Match function/entity name
            item_name = item.get("name", "") or ""
            node_id = item.get("node_id", "") or ""
            base_node_id = item.get("base_node_id", "") or ""

            name_match = (
                item_name.lower() == target_func.lower()
                or node_id.lower().endswith("::" + target_func.lower())
                or f"::{target_func.lower()}::" in node_id.lower()
                or base_node_id.lower().endswith("::" + target_func.lower())
                or f"::{target_func.lower()}::" in base_node_id.lower()
                or node_id.lower().endswith("." + target_func.lower())
                or f".{target_func.lower()}." in node_id.lower()
                or base_node_id.lower().endswith("." + target_func.lower())
                or f".{target_func.lower()}." in base_node_id.lower()
            )

            if file_match and name_match:
                return rank

        return 0

    def run(
        self,
        cases: list[RepoQACase],
        top_k: int = 10,
    ) -> RepoQAEvalReport:
        """
        Run evaluation across RepoQA *cases*.

        For each case:
        1. Run hybrid retrieval.
        2. Run vector-only retrieval (no graph).
        3. Identify matching target function rank in both modes.
        4. Assemble RepoQAQueryResult and add to report.
        """
        report = RepoQAEvalReport()
        for case in cases:
            logger.info("Evaluating RepoQA %s: %s", case.id, case.question[:60])

            # Hybrid mode retrieval
            hybrid_results = self._hybrid_retriever.retrieve(case.question, top_k=top_k)
            rank_hybrid = self._find_target_rank(hybrid_results, case)

            # Vector-only mode retrieval
            vector_results = self._hybrid_retriever.retrieve(
                case.question, top_k=top_k, skip_graph=True
            )
            rank_vector = self._find_target_rank(vector_results, case)

            qr = RepoQAQueryResult(
                query_id=case.id,
                question=case.question,
                target_function=case.target_function,
                file_path=case.file_path,
                rank_hybrid=rank_hybrid,
                rank_vector=rank_vector,
            )
            report.results.append(qr)
            logger.info(
                "RepoQA %s | hybrid rank=%d vector rank=%d | hit@1 hybrid=%.1f vector=%.1f | hit@5 hybrid=%.1f vector=%.1f | MRR hybrid=%.2f vector=%.2f",
                case.id,
                rank_hybrid,
                rank_vector,
                qr.hit_at1_hybrid,
                qr.hit_at1_vector,
                qr.hit_at5_hybrid,
                qr.hit_at5_vector,
                qr.mrr_hybrid,
                qr.mrr_vector,
            )

        return report

