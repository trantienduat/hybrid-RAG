"""
RAGAS evaluation runner — M4 #31.

Measures generation quality (faithfulness, answer_relevancy) for the hybrid
retrieval + Ollama LLM pipeline on the Q1-Q20 corpus.

Requires the [eval] extras:  pip install 'hybrid-rag[eval]'
  ragas>=0.1.0
  datasets

Usage::

    runner = RagasRunner(
        retriever=hybrid_retriever,
        ollama_url="http://localhost:11434",
        llm_model="gemma2:9b",
    )
    report = runner.run(EVAL_CORPUS[:5], top_k=20)
    print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from hybrid_rag.eval.corpus import QueryCase

if TYPE_CHECKING:
    from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert code assistant. Use ONLY the context provided to answer the question. "
    "If the context is insufficient, say so. Be concise and precise."
)


# ── LLM call ──────────────────────────────────────────────────────────────────


def _generate(ollama_url: str, model: str, question: str, context: str) -> tuple[str, float]:
    """Call Ollama /api/generate and return (answer, latency_ms)."""
    prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    t0 = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000
    return resp.json().get("response", ""), latency_ms


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class RagasSample:
    """Inputs and outputs for one evaluation sample."""

    query_id: str
    question: str
    hops: int
    query_type: str
    context: str  # concatenated chunk texts
    answer: str
    latency_ms: float
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0


@dataclass
class RagasReport:
    samples: list[RagasSample] = field(default_factory=list)

    def summary(self) -> str:
        if not self.samples:
            return "No samples evaluated."
        n = len(self.samples)
        avg_f = sum(s.faithfulness for s in self.samples) / n
        avg_ar = sum(s.answer_relevancy for s in self.samples) / n
        avg_cp = sum(s.context_precision for s in self.samples) / n
        avg_ms = sum(s.latency_ms for s in self.samples) / n
        return (
            f"RAGAS ({n} samples): "
            f"faithfulness={avg_f:.3f}  "
            f"answer_relevancy={avg_ar:.3f}  "
            f"context_precision={avg_cp:.3f}  "
            f"avg_latency={avg_ms:.0f}ms"
        )

    def as_dict(self) -> dict[str, Any]:
        n = len(self.samples)
        if not n:
            return {}

        def avg(attr: str) -> float:
            return sum(getattr(s, attr) for s in self.samples) / n

        by_hops: dict[int, dict] = {}
        for hops in (1, 2, 3):
            sub = [s for s in self.samples if s.hops == hops]
            if sub:
                by_hops[hops] = {
                    "n": len(sub),
                    "faithfulness": sum(s.faithfulness for s in sub) / len(sub),
                    "answer_relevancy": sum(s.answer_relevancy for s in sub) / len(sub),
                    "avg_latency_ms": sum(s.latency_ms for s in sub) / len(sub),
                }

        return {
            "n": n,
            "faithfulness": avg("faithfulness"),
            "answer_relevancy": avg("answer_relevancy"),
            "context_precision": avg("context_precision"),
            "avg_latency_ms": avg("latency_ms"),
            "by_hops": by_hops,
        }


# ── Runner ────────────────────────────────────────────────────────────────────


class RagasRunner:
    """
    Run RAGAS generation-quality evaluation on the Q1-Q20 corpus.

    RAGAS metrics (ragas library):
      - faithfulness        : are claims in the answer grounded in the retrieved context?
      - answer_relevancy    : is the answer responsive to the question?
      - context_precision   : are retrieved chunks actually relevant to the question?

    Falls back to neutral scores (0.5) if the ragas library is not installed or
    if the RAGAS evaluation call fails — so the pipeline always produces output.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "gemma2:9b",
    ) -> None:
        self._retriever = retriever
        self._ollama_url = ollama_url
        self._llm_model = llm_model

    # ── Public ────────────────────────────────────────────────────

    def run(
        self,
        cases: list[QueryCase],
        top_k: int = 20,
        context_n: int = 5,
    ) -> RagasReport:
        """
        For each QueryCase:
        1. Hybrid retrieval (top_k candidates).
        2. Context assembly (top context_n texts).
        3. LLM generation via Ollama.
        4. RAGAS scoring (if ragas installed).
        """
        samples: list[RagasSample] = []
        for case in cases:
            sample = self._run_one(case, top_k, context_n)
            samples.append(sample)
            logger.info(
                "%s  faithfulness=%.2f  answer_relevancy=%.2f  latency=%.0fms",
                case.id,
                sample.faithfulness,
                sample.answer_relevancy,
                sample.latency_ms,
            )

        report = RagasReport(samples=samples)
        self._score_ragas(report)
        return report

    # ── Private ───────────────────────────────────────────────────

    def _run_one(self, case: QueryCase, top_k: int, context_n: int) -> RagasSample:
        """Retrieve + generate for one query case."""
        results = self._retriever.retrieve(case.question, top_k=top_k)
        ctx_chunks = [r for r in results[:context_n] if r.get("text")]
        context = "\n\n".join(
            f"[{r.get('label', '')}] {r.get('name', '')} ({r.get('file_path', '')})\n{r['text']}"
            for r in ctx_chunks
        )
        if not context:
            context = "(no code context retrieved)"

        try:
            answer, latency_ms = _generate(
                self._ollama_url, self._llm_model, case.question, context
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed for %s: %s", case.id, exc)
            answer = ""
            latency_ms = 0.0

        return RagasSample(
            query_id=case.id,
            question=case.question,
            hops=case.hops,
            query_type=case.query_type,
            context=context,
            answer=answer,
            latency_ms=latency_ms,
        )

    def _score_ragas(self, report: RagasReport) -> None:
        """
        Score all samples using ragas library.

        Populates faithfulness / answer_relevancy / context_precision on each sample.
        On import failure or scoring error, sets neutral values (0.5).
        """
        try:
            from datasets import Dataset  # type: ignore[import]
            from ragas import evaluate  # type: ignore[import]
            from ragas.metrics import (  # type: ignore[import]
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError:
            logger.warning(
                "ragas or datasets not installed — skipping RAGAS scoring. "
                "Install with: pip install 'hybrid-rag[eval]'"
            )
            for s in report.samples:
                s.faithfulness = 0.5
                s.answer_relevancy = 0.5
                s.context_precision = 0.5
            return

        scored = [s for s in report.samples if s.answer.strip()]
        if not scored:
            return

        dataset = Dataset.from_dict(
            {
                "question": [s.question for s in scored],
                "answer": [s.answer for s in scored],
                "contexts": [[s.context] for s in scored],
            }
        )

        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
            )
            df = result.to_pandas()
            for i, s in enumerate(scored):
                s.faithfulness = float(df.iloc[i].get("faithfulness", 0.5))
                s.answer_relevancy = float(df.iloc[i].get("answer_relevancy", 0.5))
                s.context_precision = float(df.iloc[i].get("context_precision", 0.5))
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAGAS scoring failed: %s — using neutral 0.5", exc)
            for s in scored:
                s.faithfulness = 0.5
                s.answer_relevancy = 0.5
                s.context_precision = 0.5
