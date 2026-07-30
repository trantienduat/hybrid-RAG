"""
RAGAS evaluation runner — M4 #31.

Measures generation quality for the hybrid retrieval + Ollama LLM pipeline
using an explicitly configured local RAGAS judge.

Requires the [eval] extras:  pip install 'hybrid-rag[eval]'

Usage::

    runner = RagasRunner(
        retriever=hybrid_retriever,
        repository="llama-core",
        ollama_url="http://localhost:11434",
        llm_model="gemma4:12b",
    )
    report = runner.run(EVAL_CORPUS[:5], references=references, top_k=20)
    print(report.summary())
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from hybrid_rag.constants import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL
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


def _build_metrics(
    *,
    ollama_url: str,
    judge_model: str,
    embedding_model: str,
) -> tuple[Any, Any, Any]:
    """Build RAGAS 0.4 metrics with explicit local Ollama evaluators."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are unavailable; install hybrid-rag[eval]"
        ) from exc

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=f"{ollama_url.rstrip('/')}/v1",
        timeout=120.0,
        max_retries=0,
    )
    judge = llm_factory(
        judge_model,
        provider="openai",
        client=client,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(client=client, model=embedding_model)
    return (
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=1),
        ContextPrecision(llm=judge),
    )


def _format_context(result: dict[str, Any]) -> str:
    """Format code text or graph-only structural evidence for the judge."""
    name = result.get("name", "") or result.get("node_id", "")
    header = (f"[{result.get('label', '')}] {name} ({result.get('file_path', '')})").strip()
    text = str(result.get("text", "")).strip()
    if text:
        return f"{header}\n{text}"
    relation = result.get("rel", "")
    node_id = result.get("node_id", "")
    if "::" in node_id:
        owner = node_id.split("::")[-2] if len(node_id.split("::")) > 1 else ""
    else:
        owner = node_id.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in node_id else ""
    if relation == "DEFINES" and owner and name:
        structural = f"Graph evidence: {owner} defines the {result.get('label', 'entity')} {name}."
    else:
        structural = (
            f"Graph evidence: the {result.get('label', 'entity')} {name} is connected by a "
            f"{relation or 'graph'} relationship (node ID: {node_id})."
        )
    return f"{header}\n{structural}"


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class RagasSample:
    """Inputs and outputs for one evaluation sample."""

    query_id: str
    question: str
    hops: int
    query_type: str
    contexts: list[str]
    reference: str
    answer: str
    latency_ms: float
    has_graph_only_context: bool = False
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_precision_unavailable_reason: str | None = None


@dataclass
class RagasReport:
    samples: list[RagasSample] = field(default_factory=list)
    evaluator: dict[str, str] = field(default_factory=dict)

    def _average(
        self,
        attr: str,
        samples: list[RagasSample] | None = None,
        *,
        required: bool = True,
    ) -> float | None:
        rows = samples if samples is not None else self.samples
        values = [getattr(sample, attr) for sample in rows]
        if required and (not values or any(value is None for value in values)):
            raise RuntimeError(f"RAGAS metric {attr} is incomplete")
        scored = [value for value in values if value is not None]
        return sum(scored) / len(scored) if scored else None

    def summary(self) -> str:
        if not self.samples:
            return "No samples evaluated."
        avg_f = self._average("faithfulness")
        avg_ar = self._average("answer_relevancy")
        avg_cp = self._average("context_precision", required=False)
        avg_ms = sum(s.latency_ms for s in self.samples) / len(self.samples)
        cp_summary = f"{avg_cp:.3f}" if avg_cp is not None else "n/a (graph-only context)"
        return (
            f"RAGAS ({len(self.samples)} samples): "
            f"faithfulness={avg_f:.3f}  "
            f"answer_relevancy={avg_ar:.3f}  "
            f"context_precision={cp_summary}  "
            f"avg_latency={avg_ms:.0f}ms"
        )

    def as_dict(self) -> dict[str, Any]:
        if not self.samples:
            return {}

        by_hops: dict[int, dict] = {}
        for hops in (1, 2, 3):
            sub = [s for s in self.samples if s.hops == hops]
            if sub:
                by_hops[hops] = {
                    "n": len(sub),
                    "faithfulness": self._average("faithfulness", sub),
                    "answer_relevancy": self._average("answer_relevancy", sub),
                    "avg_latency_ms": sum(s.latency_ms for s in sub) / len(sub),
                }

        return {
            "n": len(self.samples),
            "evaluator": self.evaluator,
            "faithfulness": self._average("faithfulness"),
            "answer_relevancy": self._average("answer_relevancy"),
            "context_precision": self._average("context_precision", required=False),
            "context_precision_n": sum(
                sample.context_precision is not None for sample in self.samples
            ),
            "avg_latency_ms": sum(s.latency_ms for s in self.samples) / len(self.samples),
            "by_hops": by_hops,
            "samples": [
                {
                    "id": sample.query_id,
                    "question": sample.question,
                    "hops": sample.hops,
                    "query_type": sample.query_type,
                    "contexts": sample.contexts,
                    "reference": sample.reference,
                    "answer": sample.answer,
                    "latency_ms": sample.latency_ms,
                    "has_graph_only_context": sample.has_graph_only_context,
                    "faithfulness": sample.faithfulness,
                    "answer_relevancy": sample.answer_relevancy,
                    "context_precision": sample.context_precision,
                    "context_precision_unavailable_reason": sample.context_precision_unavailable_reason,
                }
                for sample in self.samples
            ],
        }


# ── Runner ────────────────────────────────────────────────────────────────────


class RagasRunner:
    """
    Run RAGAS generation-quality evaluation on the Q1-Q50 corpus.

    RAGAS 0.4 metrics:
      - faithfulness        : are claims in the answer grounded in the retrieved context?
      - answer_relevancy    : is the answer responsive to the question?
      - context_precision   : are retrieved chunks actually relevant to the question?
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        repository: str,
        ollama_url: str = "http://localhost:11434",
        llm_model: str | None = None,
        judge_model: str | None = None,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        if not repository.strip():
            raise ValueError("repository is required for RAGAS evaluation")
        self._retriever = retriever
        self._repository = repository
        self._ollama_url = ollama_url
        self._llm_model = llm_model or DEFAULT_LLM_MODEL
        self._judge_model = judge_model or self._llm_model
        self._embedding_model = embedding_model

    # ── Public ────────────────────────────────────────────────────

    def run(
        self,
        cases: list[QueryCase],
        *,
        references: dict[str, str],
        top_k: int = 20,
        context_n: int = 5,
    ) -> RagasReport:
        """
        For each QueryCase:
        1. Hybrid retrieval (top_k candidates).
        2. Context assembly (top context_n texts).
        3. LLM generation via Ollama.
        4. RAGAS scoring with explicit local judge and embeddings.
        """
        missing_references = [case.id for case in cases if not references.get(case.id, "").strip()]
        if missing_references:
            raise ValueError(f"Missing reference answers: {', '.join(missing_references)}")

        samples: list[RagasSample] = []
        for case in cases:
            sample = self._run_one(
                case,
                top_k,
                context_n,
                reference=references[case.id],
            )
            samples.append(sample)
            logger.info(
                "%s generated answer in %.0fms",
                case.id,
                sample.latency_ms,
            )

        self._score_ragas(samples)
        report = RagasReport(
            samples=samples,
            evaluator={
                "provider": "ollama",
                "answer_model": self._llm_model,
                "judge_model": self._judge_model,
                "embedding_model": self._embedding_model,
                "repository": self._repository,
            },
        )
        logger.info("RAGAS results: %s", report.summary())
        return report

    # ── Private ───────────────────────────────────────────────────

    def _run_one(
        self,
        case: QueryCase,
        top_k: int,
        context_n: int,
        *,
        reference: str,
    ) -> RagasSample:
        """Retrieve + generate for one query case."""
        results = self._retriever.retrieve(
            case.question,
            top_k=top_k,
            repository=self._repository,
        )
        contexts = [_format_context(result) for result in results[:context_n]]
        if not contexts:
            raise RuntimeError(f"No text context retrieved for {case.id}")
        context = "\n\n".join(contexts)

        try:
            answer, latency_ms = _generate(
                self._ollama_url, self._llm_model, case.question, context
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Generation failed for {case.id}: {exc}") from exc
        if not answer.strip():
            raise RuntimeError(f"Generation returned an empty answer for {case.id}")

        return RagasSample(
            query_id=case.id,
            question=case.question,
            hops=case.hops,
            query_type=case.query_type,
            contexts=contexts,
            reference=reference,
            answer=answer,
            latency_ms=latency_ms,
            has_graph_only_context=any(
                not str(result.get("text", "")).strip() for result in results[:context_n]
            ),
        )

    @staticmethod
    def _score_value(value: Any, metric: str, query_id: str) -> float:
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Invalid {metric} score for {query_id}: {score}")
        return score

    def _score_ragas(self, samples: list[RagasSample]) -> None:
        """Score samples with explicit evaluators and abort on any invalid result."""
        faithfulness, answer_relevancy, context_precision = _build_metrics(
            ollama_url=self._ollama_url,
            judge_model=self._judge_model,
            embedding_model=self._embedding_model,
        )
        asyncio.run(
            self._score_samples(
                samples,
                faithfulness,
                answer_relevancy,
                context_precision,
            )
        )

    async def _score_samples(
        self,
        samples: list[RagasSample],
        faithfulness: Any,
        answer_relevancy: Any,
        context_precision: Any,
    ) -> None:
        for sample in samples:
            try:
                faith_result = await faithfulness.ascore(
                    user_input=sample.question,
                    response=sample.answer,
                    retrieved_contexts=sample.contexts,
                )
                relevancy_result = await answer_relevancy.ascore(
                    user_input=sample.question,
                    response=sample.answer,
                )
                sample.faithfulness = self._score_value(
                    faith_result.value,
                    "faithfulness",
                    sample.query_id,
                )
                sample.answer_relevancy = self._score_value(
                    relevancy_result.value,
                    "answer_relevancy",
                    sample.query_id,
                )
                if sample.has_graph_only_context:
                    # RAGAS ContextPrecision is an LLM judgment of natural-language chunks.
                    # It is not valid for graph facts that contain no source text; retrieval
                    # precision for those cases is measured by the graph-oracle Hit@K report.
                    sample.context_precision_unavailable_reason = "graph-only structural context"
                else:
                    precision_result = await context_precision.ascore(
                        user_input=sample.question,
                        reference=sample.reference,
                        retrieved_contexts=sample.contexts,
                    )
                    sample.context_precision = self._score_value(
                        precision_result.value,
                        "context_precision",
                        sample.query_id,
                    )
            except Exception as exc:
                raise RuntimeError(f"RAGAS scoring failed for {sample.query_id}: {exc}") from exc
