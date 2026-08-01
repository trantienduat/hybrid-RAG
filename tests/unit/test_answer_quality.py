"""Validity tests for the independent answer-quality benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from hybrid_rag.eval.answer_quality import (
    AnswerQualityRunner,
    GenerationResult,
    GoldAnswerCase,
    GoldAnswerDataset,
    _write_json_atomic,
    generate_answer,
    load_gold_dataset,
)

SOURCE_IDENTITY = "sha256:" + "a" * 64
INDEX_METADATA = {
    "source_identity": SOURCE_IDENTITY,
    "index_run_id": "run-1",
    "index_schema_version": 2,
    "embedding_model": "embed",
}
ROOT = Path(__file__).resolve().parents[2]


def _payload() -> dict:
    return {
        "name": "gold-v1",
        "schema_version": 1,
        "source_identity": SOURCE_IDENTITY,
        "review": {
            "status": "draft",
            "reviewer": "",
            "reviewer_type": "",
            "reviewed_at": "",
        },
        "cases": [
            {
                "id": "AQ01",
                "difficulty": "simple",
                "question": "What happens?",
                "reference_answer": "The function returns the stored value.",
                "source_files": ["module.py"],
                "source_anchors": ["module.py:10-12"],
                "reference_contexts": ["The function returns self.value."],
            }
        ],
    }


def _write_dataset(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_json_reports_are_written_atomically(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("stale", encoding="utf-8")

    _write_json_atomic(path, {"complete": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_draft_dataset_requires_explicit_non_claiming_override(tmp_path):
    path = _write_dataset(tmp_path, _payload())

    with pytest.raises(ValueError, match="dataset is draft"):
        load_gold_dataset(path)

    dataset = load_gold_dataset(path, allow_draft=True)
    assert dataset.evidence_grade == "draft"
    assert dataset.cases[0].difficulty == "simple"
    assert dataset.cases[0].source_anchors == ("module.py:10-12",)


def test_checked_in_dataset_is_versioned_complete_and_ai_reviewed():
    dataset = load_gold_dataset(ROOT / "eval/gold/llama_index_core_answer_quality_v1.json")

    assert dataset.schema_version == 1
    assert dataset.review_status == "approved"
    assert dataset.reviewer == "Codex"
    assert dataset.reviewer_type == "ai_source_review"
    assert dataset.evidence_grade == "approved_ai_source_review"
    assert len(dataset.cases) == 30
    assert len({case.id for case in dataset.cases}) == 30
    assert {
        difficulty: sum(case.difficulty == difficulty for case in dataset.cases)
        for difficulty in ("simple", "medium", "hard")
    } == {"simple": 10, "medium": 10, "hard": 10}


def test_approved_dataset_requires_review_metadata(tmp_path):
    payload = _payload()
    payload["review"]["status"] = "approved"

    with pytest.raises(ValueError, match="require reviewer"):
        load_gold_dataset(_write_dataset(tmp_path, payload))


def test_duplicate_case_ids_are_rejected(tmp_path):
    payload = _payload()
    payload["cases"].append(dict(payload["cases"][0]))

    with pytest.raises(ValueError, match="unique"):
        load_gold_dataset(_write_dataset(tmp_path, payload), allow_draft=True)


def test_invalid_case_difficulty_is_rejected(tmp_path):
    payload = _payload()
    payload["cases"][0]["difficulty"] = "extreme"

    with pytest.raises(ValueError, match="difficulty"):
        load_gold_dataset(_write_dataset(tmp_path, payload), allow_draft=True)


def test_source_anchor_must_reference_a_declared_file_and_line_range(tmp_path):
    payload = _payload()
    payload["cases"][0]["source_anchors"] = ["other.py:12-10"]

    with pytest.raises(ValueError, match="source anchor"):
        load_gold_dataset(_write_dataset(tmp_path, payload), allow_draft=True)


def test_generate_answer_requires_exact_ollama_usage(monkeypatch):
    response = MagicMock()
    response.json.return_value = {
        "response": "An answer.",
        "prompt_eval_count": 42,
        "eval_count": 7,
    }
    response.raise_for_status.return_value = None
    client = MagicMock()
    client.__enter__.return_value.post.side_effect = [
        httpx.ReadTimeout("slow local model"),
        response,
    ]
    monkeypatch.setattr(
        "hybrid_rag.eval.answer_quality.httpx.Client",
        lambda **_kwargs: client,
    )

    result = generate_answer(
        ollama_url="http://ollama",
        model="answer-model",
        question="Question?",
        contexts=["Context."],
        seed=9,
    )

    assert result.answer == "An answer."
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 7
    request = client.__enter__.return_value.post.call_args
    assert request.kwargs["json"]["options"] == {"temperature": 0, "seed": 9}
    assert client.__enter__.return_value.post.call_count == 2


def test_runner_compares_modes_and_resumes_atomic_checkpoint(monkeypatch, tmp_path):
    dataset = GoldAnswerDataset(
        name="gold-v1",
        schema_version=1,
        source_identity=SOURCE_IDENTITY,
        review_status="approved",
        reviewer="reviewer",
        reviewer_type="human",
        reviewed_at="2026-07-30",
        cases=(
            GoldAnswerCase(
                id="AQ01",
                difficulty="simple",
                question="What happens?",
                reference_answer="It returns a value.",
                source_files=("module.py",),
                source_anchors=("module.py:10-12",),
                reference_contexts=("return self.value",),
            ),
        ),
    )
    retriever = MagicMock()
    retriever.retrieve.side_effect = lambda _question, skip_graph, **_kwargs: (
        [{"name": "target", "file_path": "module.py", "text": "return self.value"}]
        if not skip_graph
        else [{"name": "other", "file_path": "other.py", "text": "return None"}]
    )
    monkeypatch.setattr(
        "hybrid_rag.eval.answer_quality.generate_answer",
        lambda **_kwargs: GenerationResult("It returns a value.", 12.0, 50, 8),
    )

    class Metric:
        async def ascore(self, **_kwargs):
            return MagicMock(value=0.8)

    class Client:
        closed = False

        async def close(self):
            self.closed = True

    client = Client()
    monkeypatch.setattr(
        "hybrid_rag.eval.answer_quality._build_answer_metrics",
        lambda **_kwargs: (Metric(), Metric(), Metric(), client),
    )
    runner = AnswerQualityRunner(
        retriever,
        repository="repo",
        ollama_url="http://ollama",
        answer_model="answer",
        judge_model="judge",
        embedding_model="embed",
        input_cost_per_million=2.0,
        output_cost_per_million=4.0,
    )

    with pytest.raises(ValueError, match="top_k"):
        runner.run(
            dataset,
            index_metadata=INDEX_METADATA,
            top_k=0,
        )
    with pytest.raises(ValueError, match="context_n"):
        runner.run(
            dataset,
            index_metadata=INDEX_METADATA,
            top_k=5,
            context_n=6,
        )

    checkpoint = tmp_path / "benchmark.checkpoint.json"
    report = runner.run(
        dataset,
        index_metadata=INDEX_METADATA,
        repeats=2,
        checkpoint_path=checkpoint,
    )
    result = report.as_dict()

    assert len(report.records) == 4
    assert client.closed is True
    assert result["summary"]["hybrid"]["source_hit_rate"] == 1.0
    assert result["summary"]["vector"]["source_hit_rate"] == 0.0
    assert result["summary"]["hybrid"]["answer_correctness"] == 0.8
    assert result["summary"]["hybrid"]["avg_prompt_tokens"] == 50.0
    assert result["summary"]["hybrid"]["total_estimated_api_cost_usd"] == pytest.approx(0.000264)
    assert result["summary"]["hybrid"]["avg_estimated_api_cost_per_answer_usd"] == pytest.approx(
        0.000132
    )
    assert result["config"]["judge_independent"] is True
    assert result["dataset"]["difficulty_counts"] == {
        "simple": 1,
        "medium": 0,
        "hard": 0,
    }
    assert result["summary"]["delta_hybrid_minus_vector"]["source_hit_rate"] == 1.0
    assert {record.mode for record in report.records} == {"hybrid", "vector"}
    assert checkpoint.is_file()
    assert not checkpoint.with_name(f"{checkpoint.name}.tmp").exists()
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["schema_version"] == 2
    assert checkpoint_payload["index"] == INDEX_METADATA

    retrieval_calls = retriever.retrieve.call_count
    resumed = runner.run(
        dataset,
        index_metadata=INDEX_METADATA,
        repeats=2,
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert len(resumed.records) == 4
    assert retriever.retrieve.call_count == retrieval_calls

    corrupt = json.loads(checkpoint.read_text(encoding="utf-8"))
    corrupt["records"][0]["faithfulness"] = float("nan")
    checkpoint.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid faithfulness score"):
        runner.run(
            dataset,
            index_metadata=INDEX_METADATA,
            repeats=2,
            checkpoint_path=checkpoint,
            resume=True,
        )

    with pytest.raises(ValueError, match="Checkpoint configuration"):
        runner.run(
            dataset,
            index_metadata=INDEX_METADATA,
            repeats=1,
            checkpoint_path=checkpoint,
            resume=True,
        )


@pytest.mark.parametrize(
    ("input_cost", "output_cost"),
    [
        (1.0, None),
        (None, 1.0),
        (-1.0, 1.0),
        (1.0, float("inf")),
    ],
)
def test_runner_rejects_incomplete_or_invalid_cost_rates(input_cost, output_cost):
    with pytest.raises(ValueError, match="cost rates"):
        AnswerQualityRunner(
            MagicMock(),
            repository="repo",
            ollama_url="http://ollama",
            answer_model="answer",
            judge_model="judge",
            embedding_model="embed",
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
        )


def test_runner_rejects_mismatched_source_snapshot():
    dataset = GoldAnswerDataset(
        name="gold-v1",
        schema_version=1,
        source_identity=SOURCE_IDENTITY,
        review_status="approved",
        reviewer="reviewer",
        reviewer_type="human",
        reviewed_at="2026-07-30",
        cases=(),
    )
    runner = AnswerQualityRunner(
        MagicMock(),
        repository="repo",
        ollama_url="http://ollama",
        answer_model="answer",
        judge_model="judge",
        embedding_model="embed",
    )

    with pytest.raises(ValueError, match="does not match"):
        runner.run(dataset, index_metadata={**INDEX_METADATA, "source_identity": "sha256:other"})


def test_runner_requires_complete_index_provenance():
    dataset = GoldAnswerDataset(
        name="gold-v1",
        schema_version=1,
        source_identity=SOURCE_IDENTITY,
        review_status="approved",
        reviewer="reviewer",
        reviewer_type="human",
        reviewed_at="2026-07-30",
        cases=(),
    )
    runner = AnswerQualityRunner(
        MagicMock(),
        repository="repo",
        ollama_url="http://ollama",
        answer_model="answer",
        judge_model="judge",
        embedding_model="embed",
    )

    with pytest.raises(ValueError, match="index_run_id"):
        runner.run(dataset, index_metadata={"source_identity": SOURCE_IDENTITY})
