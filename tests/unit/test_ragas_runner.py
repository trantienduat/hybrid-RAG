"""Validity tests for the RAGAS answer-quality runner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hybrid_rag.eval.corpus import QueryCase
from hybrid_rag.eval.ragas_runner import RagasReport, RagasRunner, RagasSample, _build_metrics


@pytest.fixture
def case() -> QueryCase:
    return QueryCase(
        id="Q1",
        question="What does BaseRetriever define?",
        hops=1,
        query_type="structural",
        ground_truth_cypher="RETURN 'retrieve'",
    )


def test_requires_reference_for_every_case(case):
    runner = RagasRunner(MagicMock(), repository="llama-core")

    with pytest.raises(ValueError, match="Missing reference answers: Q1"):
        runner.run([case], references={})


def test_build_metrics_uses_pinned_ragas_collections():
    metrics = _build_metrics(
        ollama_url="http://localhost:11434",
        judge_model="gemma4:12b",
        embedding_model="nomic-embed-text",
    )

    assert [type(metric).__name__ for metric in metrics] == [
        "Faithfulness",
        "AnswerRelevancy",
        "ContextPrecision",
    ]
    assert metrics[0].llm.is_async is True
    assert metrics[1].embeddings.is_async is True
    assert metrics[1].strictness == 1


def test_retrieval_is_repository_scoped(monkeypatch, case):
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "name": "retrieve",
            "label": "Function",
            "file_path": "retriever.py",
            "text": "def retrieve(): pass",
        }
    ]
    monkeypatch.setattr(
        "hybrid_rag.eval.ragas_runner._generate",
        lambda *_args: ("It defines retrieve.", 12.0),
    )
    runner = RagasRunner(retriever, repository="llama-core")

    sample = runner._run_one(case, top_k=20, context_n=5, reference="It defines retrieve.")

    retriever.retrieve.assert_called_once_with(
        case.question,
        top_k=20,
        repository="llama-core",
    )
    assert sample.reference == "It defines retrieve."
    assert sample.contexts == [
        "[Function] retrieve (retriever.py)\ndef retrieve(): pass",
    ]


def test_generation_failure_aborts_evaluation(monkeypatch, case):
    retriever = MagicMock()
    retriever.retrieve.return_value = [{"text": "code"}]

    def fail_generation(*_args):
        raise TimeoutError("Ollama timed out")

    monkeypatch.setattr("hybrid_rag.eval.ragas_runner._generate", fail_generation)
    runner = RagasRunner(retriever, repository="llama-core")

    with pytest.raises(RuntimeError, match="Generation failed for Q1"):
        runner._run_one(case, top_k=20, context_n=5, reference="Expected answer")


def test_graph_only_results_are_preserved_as_context(monkeypatch, case):
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "node_id": "retriever.py::BaseRetriever::retrieve",
            "name": "retrieve",
            "label": "Function",
            "file_path": "retriever.py",
            "source": "graph",
            "rel": "DEFINES",
            "text": "",
        }
    ]
    monkeypatch.setattr(
        "hybrid_rag.eval.ragas_runner._generate",
        lambda *_args: ("It defines retrieve.", 12.0),
    )
    runner = RagasRunner(retriever, repository="llama-core")

    sample = runner._run_one(case, top_k=20, context_n=5, reference="It defines retrieve.")

    assert sample.contexts == [
        "[Function] retrieve (retriever.py)\n"
        "Graph evidence: BaseRetriever defines the Function retrieve."
    ]


def test_scoring_failure_does_not_create_neutral_scores(monkeypatch):
    class FailingMetric:
        async def ascore(self, **_kwargs):
            raise RuntimeError("judge unavailable")

    monkeypatch.setattr(
        "hybrid_rag.eval.ragas_runner._build_metrics",
        lambda **_kwargs: (FailingMetric(), FailingMetric(), FailingMetric()),
    )
    sample = RagasSample(
        query_id="Q1",
        question="Question",
        hops=1,
        query_type="structural",
        contexts=["context"],
        reference="reference",
        answer="answer",
        latency_ms=10.0,
    )
    runner = RagasRunner(MagicMock(), repository="llama-core")

    with pytest.raises(RuntimeError, match="RAGAS scoring failed for Q1"):
        runner._score_ragas([sample])

    assert sample.faithfulness is None
    assert sample.answer_relevancy is None
    assert sample.context_precision is None


def test_scoring_uses_context_answer_and_reference(monkeypatch):
    calls = {}

    class Metric:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        async def ascore(self, **kwargs):
            calls[self.name] = kwargs
            return MagicMock(value=self.value)

    monkeypatch.setattr(
        "hybrid_rag.eval.ragas_runner._build_metrics",
        lambda **_kwargs: (
            Metric("faithfulness", 0.9),
            Metric("answer_relevancy", 0.8),
            Metric("context_precision", 0.7),
        ),
    )
    sample = RagasSample(
        query_id="Q1",
        question="Question",
        hops=1,
        query_type="structural",
        contexts=["first", "second"],
        reference="Expected answer",
        answer="Generated answer",
        latency_ms=10.0,
    )
    runner = RagasRunner(MagicMock(), repository="llama-core")

    runner._score_ragas([sample])

    assert sample.faithfulness == 0.9
    assert sample.answer_relevancy == 0.8
    assert sample.context_precision == 0.7
    assert calls["faithfulness"] == {
        "user_input": "Question",
        "response": "Generated answer",
        "retrieved_contexts": ["first", "second"],
    }
    assert calls["answer_relevancy"] == {
        "user_input": "Question",
        "response": "Generated answer",
    }
    assert calls["context_precision"] == {
        "user_input": "Question",
        "reference": "Expected answer",
        "retrieved_contexts": ["first", "second"],
    }


def test_graph_only_context_does_not_report_invalid_context_precision(monkeypatch):
    class Metric:
        async def ascore(self, **_kwargs):
            return MagicMock(value=0.9)

    monkeypatch.setattr(
        "hybrid_rag.eval.ragas_runner._build_metrics",
        lambda **_kwargs: (Metric(), Metric(), Metric()),
    )
    sample = RagasSample(
        query_id="Q1",
        question="Question",
        hops=1,
        query_type="structural",
        contexts=["Graph evidence"],
        reference="Reference",
        answer="Answer",
        latency_ms=10.0,
        has_graph_only_context=True,
    )
    runner = RagasRunner(MagicMock(), repository="llama-core")

    runner._score_ragas([sample])

    assert sample.context_precision is None
    assert sample.context_precision_unavailable_reason == "graph-only structural context"
    report = RagasReport(samples=[sample])
    assert report.as_dict()["context_precision"] is None
    assert report.as_dict()["context_precision_n"] == 0
