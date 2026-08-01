"""Independent answer-quality benchmark primitives."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from hybrid_rag.config import validate_local_ollama_url
from hybrid_rag.eval.ragas_runner import _SYSTEM_PROMPT, _format_context

logger = logging.getLogger(__name__)

_CHECKPOINT_INDEX_FIELDS = (
    "source_identity",
    "index_run_id",
    "index_schema_version",
    "embedding_model",
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically so interruptions preserve the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class GoldAnswerCase:
    """One source-reviewed answer-quality case."""

    id: str
    difficulty: str
    question: str
    reference_answer: str
    source_files: tuple[str, ...]
    source_anchors: tuple[str, ...]
    reference_contexts: tuple[str, ...]


@dataclass(frozen=True)
class GoldAnswerDataset:
    """Versioned gold cases tied to one immutable source snapshot."""

    name: str
    schema_version: int
    source_identity: str
    review_status: str
    reviewer: str
    reviewer_type: str
    reviewed_at: str
    cases: tuple[GoldAnswerCase, ...]

    @property
    def evidence_grade(self) -> str:
        if self.review_status != "approved":
            return "draft"
        if self.reviewer_type == "human":
            return "approved_human_review"
        return "approved_ai_source_review"


def load_gold_dataset(path: Path | str, *, allow_draft: bool = False) -> GoldAnswerDataset:
    """Load and validate an answer-quality dataset."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Gold dataset not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))

    if payload.get("schema_version") != 1:
        raise ValueError("Gold dataset schema_version must be 1")
    source_identity = str(payload.get("source_identity", "")).strip()
    if not source_identity.startswith(("sha256:", "git:")):
        raise ValueError("Gold dataset requires an immutable source_identity")

    review = payload.get("review", {})
    review_status = str(review.get("status", "")).strip()
    if review_status not in {"draft", "approved"}:
        raise ValueError("Gold dataset review.status must be draft or approved")
    reviewer = str(review.get("reviewer", "")).strip()
    reviewer_type = str(review.get("reviewer_type", "")).strip()
    reviewed_at = str(review.get("reviewed_at", "")).strip()
    if review_status == "approved" and (
        not reviewer or reviewer_type not in {"human", "ai_source_review"} or not reviewed_at
    ):
        raise ValueError(
            "Approved gold datasets require reviewer, reviewer_type "
            "(human or ai_source_review), and reviewed_at"
        )
    if review_status != "approved" and not allow_draft:
        raise ValueError(
            "Gold dataset is draft; obtain independent review or pass allow_draft=True "
            "for non-claiming development runs"
        )

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Gold dataset requires at least one case")

    cases: list[GoldAnswerCase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        case_id = str(raw.get("id", "")).strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Gold case IDs must be non-empty and unique: {case_id!r}")
        seen_ids.add(case_id)
        difficulty = str(raw.get("difficulty", "")).strip()
        if difficulty not in {"simple", "medium", "hard"}:
            raise ValueError(f"Gold case {case_id} difficulty must be simple, medium, or hard")
        required_text = {
            key: str(raw.get(key, "")).strip() for key in ("question", "reference_answer")
        }
        if not all(required_text.values()):
            raise ValueError(f"Gold case {case_id} has an empty question or reference answer")

        sequence_fields: dict[str, tuple[str, ...]] = {}
        for key in ("source_files", "source_anchors", "reference_contexts"):
            values = raw.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"Gold case {case_id} requires non-empty {key}")
            normalized = tuple(str(value).strip() for value in values)
            if any(not value for value in normalized):
                raise ValueError(f"Gold case {case_id} contains an empty {key} value")
            sequence_fields[key] = normalized

        for anchor in sequence_fields["source_anchors"]:
            match = re.fullmatch(r"(.+):(\d+)-(\d+)", anchor)
            if (
                match is None
                or match.group(1) not in sequence_fields["source_files"]
                or int(match.group(3)) < int(match.group(2))
            ):
                raise ValueError(
                    f"Gold case {case_id} has invalid or undeclared source anchor: {anchor}"
                )

        cases.append(
            GoldAnswerCase(
                id=case_id,
                difficulty=difficulty,
                question=required_text["question"],
                reference_answer=required_text["reference_answer"],
                source_files=sequence_fields["source_files"],
                source_anchors=sequence_fields["source_anchors"],
                reference_contexts=sequence_fields["reference_contexts"],
            )
        )

    return GoldAnswerDataset(
        name=str(payload.get("name", "")).strip() or source.stem,
        schema_version=1,
        source_identity=source_identity,
        review_status=review_status,
        reviewer=reviewer,
        reviewer_type=reviewer_type,
        reviewed_at=reviewed_at,
        cases=tuple(cases),
    )


@dataclass(frozen=True)
class GenerationResult:
    """Answer and exact Ollama usage counters."""

    answer: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def generate_answer(
    *,
    ollama_url: str,
    model: str,
    question: str,
    contexts: list[str],
    seed: int,
) -> GenerationResult:
    """Generate one deterministic answer and require exact Ollama token counters."""
    ollama_url = validate_local_ollama_url(ollama_url)
    context = "\n\n".join(contexts) if contexts else "(no code context retrieved)"
    prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    started = time.perf_counter()
    response: httpx.Response | None = None
    with httpx.Client(timeout=600.0) as client:
        for attempt in range(1, 4):
            try:
                response = client.post(
                    f"{ollama_url.rstrip('/')}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0, "seed": seed},
                    },
                )
                response.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 3:
                    raise
                logger.warning(
                    "Answer generation attempt %d/3 failed; retrying",
                    attempt,
                    exc_info=True,
                )
    assert response is not None
    latency_ms = (time.perf_counter() - started) * 1000
    payload = response.json()
    answer = str(payload.get("response", "")).strip()
    prompt_tokens = int(payload.get("prompt_eval_count", 0))
    completion_tokens = int(payload.get("eval_count", 0))
    if not answer:
        raise RuntimeError("Answer generation returned empty text")
    if prompt_tokens <= 0 or completion_tokens <= 0:
        raise RuntimeError("Ollama response did not include positive exact token counters")
    return GenerationResult(answer, latency_ms, prompt_tokens, completion_tokens)


@dataclass
class AnswerQualityRecord:
    """Raw result for one case, mode, and repeat."""

    case_id: str
    question: str
    reference_answer: str
    source_files: list[str]
    run: int
    mode: str
    contexts: list[str]
    retrieved_files: list[str]
    answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    source_hit: float
    estimated_api_cost_usd: float | None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_correctness: float | None = None


def _build_answer_metrics(
    *,
    ollama_url: str,
    judge_model: str,
    embedding_model: str,
) -> tuple[Any, Any, Any, Any]:
    """Build answer metrics and return their shared async client for cleanup."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, Faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "Answer-quality dependencies are unavailable; install hybrid-rag[eval]"
        ) from exc

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=f"{ollama_url.rstrip('/')}/v1",
        timeout=600.0,
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
        AnswerCorrectness(llm=judge, embeddings=embeddings),
        client,
    )


def _valid_score(value: Any, metric: str, record: AnswerQualityRecord) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(
            f"Invalid {metric} score for {record.case_id}/{record.mode}/run-{record.run}: {score}"
        )
    return score


@dataclass
class AnswerQualityReport:
    """Benchmark metadata, raw records, and aggregate comparison."""

    dataset: GoldAnswerDataset
    repository: str
    answer_model: str
    judge_model: str
    embedding_model: str
    top_k: int
    context_n: int
    repeats: int
    seed: int
    index_metadata: dict[str, Any]
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    records: list[AnswerQualityRecord] = field(default_factory=list)

    @staticmethod
    def _mean(records: list[AnswerQualityRecord], attr: str) -> float:
        values = [float(getattr(record, attr)) for record in records]
        return statistics.fmean(values) if values else 0.0

    def checkpoint_dict(self) -> dict[str, Any]:
        """Serialize resumable raw state without requiring completed scores."""
        return {
            "schema_version": 2,
            "dataset": {
                "name": self.dataset.name,
                "source_identity": self.dataset.source_identity,
                "case_ids": [case.id for case in self.dataset.cases],
            },
            "config": {
                "repository": self.repository,
                "answer_model": self.answer_model,
                "judge_model": self.judge_model,
                "embedding_model": self.embedding_model,
                "top_k": self.top_k,
                "context_n": self.context_n,
                "repeats": self.repeats,
                "seed": self.seed,
                "input_cost_per_million": self.input_cost_per_million,
                "output_cost_per_million": self.output_cost_per_million,
            },
            "index": {field: self.index_metadata.get(field) for field in _CHECKPOINT_INDEX_FIELDS},
            "records": [asdict(record) for record in self.records],
        }

    def _mode_summary(self, mode: str) -> dict[str, Any]:
        records = [record for record in self.records if record.mode == mode]
        costs = [record.estimated_api_cost_usd for record in records]
        total_cost = (
            sum(float(cost) for cost in costs)
            if costs and all(cost is not None for cost in costs)
            else None
        )
        return {
            "n": len(records),
            "source_hit_rate": self._mean(records, "source_hit"),
            "faithfulness": self._mean(records, "faithfulness"),
            "answer_relevancy": self._mean(records, "answer_relevancy"),
            "answer_correctness": self._mean(records, "answer_correctness"),
            "avg_retrieval_latency_ms": self._mean(records, "retrieval_latency_ms"),
            "avg_generation_latency_ms": self._mean(records, "generation_latency_ms"),
            "avg_prompt_tokens": self._mean(records, "prompt_tokens"),
            "avg_completion_tokens": self._mean(records, "completion_tokens"),
            "total_estimated_api_cost_usd": total_cost,
            "avg_estimated_api_cost_per_answer_usd": (
                total_cost / len(records) if total_cost is not None else None
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        hybrid = self._mode_summary("hybrid")
        vector = self._mode_summary("vector")
        delta_keys = (
            "source_hit_rate",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "avg_retrieval_latency_ms",
            "avg_generation_latency_ms",
            "avg_prompt_tokens",
            "avg_completion_tokens",
        )
        return {
            "schema_version": 1,
            "evidence_grade": self.dataset.evidence_grade,
            "dataset": {
                "name": self.dataset.name,
                "source_identity": self.dataset.source_identity,
                "review_status": self.dataset.review_status,
                "reviewer": self.dataset.reviewer,
                "reviewer_type": self.dataset.reviewer_type,
                "reviewed_at": self.dataset.reviewed_at,
                "cases": len(self.dataset.cases),
                "difficulty_counts": {
                    difficulty: sum(case.difficulty == difficulty for case in self.dataset.cases)
                    for difficulty in ("simple", "medium", "hard")
                },
            },
            "config": {
                "repository": self.repository,
                "answer_model": self.answer_model,
                "judge_model": self.judge_model,
                "judge_independent": self.answer_model != self.judge_model,
                "embedding_model": self.embedding_model,
                "top_k": self.top_k,
                "context_n": self.context_n,
                "repeats": self.repeats,
                "seed": self.seed,
                "input_cost_per_million": self.input_cost_per_million,
                "output_cost_per_million": self.output_cost_per_million,
            },
            "index_metadata": self.index_metadata,
            "summary": {
                "hybrid": hybrid,
                "vector": vector,
                "delta_hybrid_minus_vector": {
                    **{key: hybrid[key] - vector[key] for key in delta_keys},
                    "avg_estimated_api_cost_per_answer_usd": (
                        hybrid["avg_estimated_api_cost_per_answer_usd"]
                        - vector["avg_estimated_api_cost_per_answer_usd"]
                        if hybrid["avg_estimated_api_cost_per_answer_usd"] is not None
                        and vector["avg_estimated_api_cost_per_answer_usd"] is not None
                        else None
                    ),
                },
            },
            "records": [asdict(record) for record in self.records],
        }


class AnswerQualityRunner:
    """Run identical answer generation and scoring for Hybrid and vector retrieval."""

    def __init__(
        self,
        retriever: Any,
        *,
        repository: str,
        ollama_url: str,
        answer_model: str,
        judge_model: str,
        embedding_model: str,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        self._retriever = retriever
        self._repository = repository
        self._ollama_url = validate_local_ollama_url(ollama_url)
        self._answer_model = answer_model
        self._judge_model = judge_model
        self._embedding_model = embedding_model
        rates = (input_cost_per_million, output_cost_per_million)
        if (rates[0] is None) != (rates[1] is None) or any(
            rate is not None and (not math.isfinite(rate) or rate < 0) for rate in rates
        ):
            raise ValueError(
                "Input and output cost rates must both be omitted or both be finite "
                "non-negative values"
            )
        self._input_cost = input_cost_per_million
        self._output_cost = output_cost_per_million

    def _estimated_cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self._input_cost is None or self._output_cost is None:
            return None
        return (
            prompt_tokens * self._input_cost + completion_tokens * self._output_cost
        ) / 1_000_000

    @staticmethod
    def _save_checkpoint(report: AnswerQualityReport, path: Path) -> None:
        """Atomically persist resumable benchmark state."""
        _write_json_atomic(path, report.checkpoint_dict())

    @staticmethod
    def _load_checkpoint(report: AnswerQualityReport, path: Path) -> None:
        """Load a checkpoint only when all immutable run settings match."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = report.checkpoint_dict()
        raw_records = payload.pop("records", None)
        expected.pop("records")
        if payload != expected:
            raise ValueError(f"Checkpoint configuration does not match this benchmark run: {path}")
        if not isinstance(raw_records, list):
            raise ValueError(f"Checkpoint records must be a list: {path}")

        cases = {case.id: case for case in report.dataset.cases}
        expected_keys = {
            (case.id, run_index, mode)
            for run_index in range(1, report.repeats + 1)
            for case in report.dataset.cases
            for mode in ("hybrid", "vector")
        }
        seen: set[tuple[str, int, str]] = set()
        records: list[AnswerQualityRecord] = []
        for raw_record in raw_records:
            try:
                record = AnswerQualityRecord(**raw_record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid checkpoint record in {path}") from exc
            key = (record.case_id, record.run, record.mode)
            case = cases.get(record.case_id)
            scores = (
                record.faithfulness,
                record.answer_relevancy,
                record.answer_correctness,
            )
            partially_scored = any(score is None for score in scores) and any(
                score is not None for score in scores
            )
            if (
                key not in expected_keys
                or key in seen
                or case is None
                or record.question != case.question
                or record.reference_answer != case.reference_answer
                or record.source_files != list(case.source_files)
                or partially_scored
            ):
                raise ValueError(f"Checkpoint contains inconsistent record {key}: {path}")
            if scores[0] is not None:
                try:
                    record.faithfulness = _valid_score(scores[0], "faithfulness", record)
                    record.answer_relevancy = _valid_score(scores[1], "answer_relevancy", record)
                    record.answer_correctness = _valid_score(
                        scores[2], "answer_correctness", record
                    )
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError(str(exc)) from exc
            seen.add(key)
            records.append(record)
        report.records = records

    def run(
        self,
        dataset: GoldAnswerDataset,
        *,
        index_metadata: dict[str, Any],
        top_k: int = 20,
        context_n: int = 5,
        repeats: int = 3,
        seed: int = 42,
        checkpoint_path: Path | None = None,
        resume: bool = False,
    ) -> AnswerQualityReport:
        if index_metadata.get("source_identity") != dataset.source_identity:
            raise ValueError(
                "Gold dataset source_identity does not match the indexed source snapshot"
            )
        missing_index_fields = [
            field for field in _CHECKPOINT_INDEX_FIELDS if index_metadata.get(field) in (None, "")
        ]
        if missing_index_fields:
            raise ValueError(
                "Answer-quality benchmark requires index metadata: "
                + ", ".join(missing_index_fields)
            )
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if context_n < 1 or context_n > top_k:
            raise ValueError("context_n must be between 1 and top_k")

        report = AnswerQualityReport(
            dataset=dataset,
            repository=self._repository,
            answer_model=self._answer_model,
            judge_model=self._judge_model,
            embedding_model=self._embedding_model,
            top_k=top_k,
            context_n=context_n,
            repeats=repeats,
            seed=seed,
            index_metadata=index_metadata,
            input_cost_per_million=self._input_cost,
            output_cost_per_million=self._output_cost,
        )
        if resume and checkpoint_path is not None and checkpoint_path.is_file():
            self._load_checkpoint(report, checkpoint_path)
            logger.info(
                "Resumed %d checkpointed records from %s",
                len(report.records),
                checkpoint_path,
            )

        completed = {(record.case_id, record.run, record.mode) for record in report.records}
        for run_index in range(1, repeats + 1):
            modes = ("hybrid", "vector") if run_index % 2 else ("vector", "hybrid")
            for case in dataset.cases:
                for mode in modes:
                    if (case.id, run_index, mode) in completed:
                        continue
                    logger.info(
                        "Generating %s run=%d mode=%s",
                        case.id,
                        run_index,
                        mode,
                    )
                    started = time.perf_counter()
                    results = self._retriever.retrieve(
                        case.question,
                        top_k=top_k,
                        skip_graph=mode == "vector",
                        repository=self._repository,
                    )
                    retrieval_latency_ms = (time.perf_counter() - started) * 1000
                    selected = results[:context_n]
                    contexts = [_format_context(result) for result in selected]
                    retrieved_files = [
                        str(result.get("file_path", "")).strip()
                        for result in selected
                        if str(result.get("file_path", "")).strip()
                    ]
                    generation = generate_answer(
                        ollama_url=self._ollama_url,
                        model=self._answer_model,
                        question=case.question,
                        contexts=contexts,
                        seed=seed + run_index - 1,
                    )
                    report.records.append(
                        AnswerQualityRecord(
                            case_id=case.id,
                            question=case.question,
                            reference_answer=case.reference_answer,
                            source_files=list(case.source_files),
                            run=run_index,
                            mode=mode,
                            contexts=contexts,
                            retrieved_files=retrieved_files,
                            answer=generation.answer,
                            retrieval_latency_ms=retrieval_latency_ms,
                            generation_latency_ms=generation.latency_ms,
                            prompt_tokens=generation.prompt_tokens,
                            completion_tokens=generation.completion_tokens,
                            source_hit=float(
                                any(path in case.source_files for path in retrieved_files)
                            ),
                            estimated_api_cost_usd=self._estimated_cost(
                                generation.prompt_tokens,
                                generation.completion_tokens,
                            ),
                        )
                    )
                    if checkpoint_path is not None:
                        self._save_checkpoint(report, checkpoint_path)

        logger.info("Scoring %d generated answers", len(report.records))
        self._score(report, checkpoint_path)
        return report

    def _score(
        self,
        report: AnswerQualityReport,
        checkpoint_path: Path | None,
    ) -> None:
        asyncio.run(self._score_async(report, checkpoint_path))

    async def _score_async(
        self,
        report: AnswerQualityReport,
        checkpoint_path: Path | None,
    ) -> None:
        faithfulness, relevancy, correctness, client = _build_answer_metrics(
            ollama_url=self._ollama_url,
            judge_model=self._judge_model,
            embedding_model=self._embedding_model,
        )
        try:
            for record in report.records:
                if (
                    record.faithfulness is not None
                    and record.answer_relevancy is not None
                    and record.answer_correctness is not None
                ):
                    continue
                logger.info(
                    "Scoring %s run=%d mode=%s",
                    record.case_id,
                    record.run,
                    record.mode,
                )
                contexts = record.contexts or ["(no code context retrieved)"]
                faith_result = await faithfulness.ascore(
                    user_input=record.question,
                    response=record.answer,
                    retrieved_contexts=contexts,
                )
                relevance_result = await relevancy.ascore(
                    user_input=record.question,
                    response=record.answer,
                )
                correctness_result = await correctness.ascore(
                    user_input=record.question,
                    response=record.answer,
                    reference=record.reference_answer,
                )
                record.faithfulness = _valid_score(faith_result.value, "faithfulness", record)
                record.answer_relevancy = _valid_score(
                    relevance_result.value, "answer_relevancy", record
                )
                record.answer_correctness = _valid_score(
                    correctness_result.value, "answer_correctness", record
                )
                if checkpoint_path is not None:
                    self._save_checkpoint(report, checkpoint_path)
        except Exception as exc:
            raise RuntimeError(f"Answer-quality scoring failed: {exc}") from exc
        finally:
            await client.close()
