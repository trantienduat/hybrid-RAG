"""Regression gates for historical and regenerated answer-quality datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hybrid_rag.eval.gold_validation import (
    assert_preserved_cases,
    validate_case_quality,
    validate_distribution,
    validate_review_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "eval" / "gold"
FIXTURE_ROOT = ROOT / "fixtures" / "small_repo"

HISTORICAL_SHA256 = {
    "llama_index_core_answer_quality_v1.json": "c11a4d3fb7dd9645e81a8e060a5c0a719fd3b005b81c9827677717918f03ec37",
    "transformers_v5_9_answer_quality_v1.json": "066914fed173337ee78c1f644348c252091275101faad2d8b414bb15fcf4c894",
    "langchain_core_v1_4_7_answer_quality_v1.json": "8f6e61fd2bf1f22bfd8adaa5513283d35a4a68ca2f2df382e10de1cf8b718d95",
}


def case(
    question: str,
    *,
    case_id: str = "AQ01",
    anchor: str = "math_utils.py:7-9",
    file: str = "math_utils.py",
) -> dict:
    return {
        "id": case_id,
        "difficulty": "simple",
        "question": question,
        "reference_answer": "The function returns the documented result.",
        "source_files": [file],
        "source_anchors": [anchor],
        "reference_contexts": ["The source documents the function behavior."],
    }


def test_historical_gold_files_are_unchanged():
    for filename, expected in HISTORICAL_SHA256.items():
        payload = (GOLD_DIR / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_rejects_numbered_component_template():
    cases = [
        case("How does component #11 work?", case_id="AQ01"),
        case("How does component #12 work?", case_id="AQ02"),
    ]

    with pytest.raises(ValueError, match="normalized question"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_generic_component_name_template():
    cases = [case("How does the component work?")]

    with pytest.raises(ValueError, match="normalized question"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_duplicate_normalized_questions():
    cases = [
        case("What does Widget 1 return?", case_id="AQ01"),
        case("what does widget 2 return?", case_id="AQ02"),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_invalid_anchor_bounds():
    candidate = case("What does Widget.run return?", anchor="module.py:10-99")

    with pytest.raises(ValueError, match="anchor"):
        validate_case_quality([candidate], source_root=FIXTURE_ROOT)


def test_rejects_source_file_concentration():
    cases = [
        case(
            f"What does symbol {chr(ord('a') + i)} do?",
            case_id=f"AQ{i:02d}",
        )
        for i in range(16)
    ]

    with pytest.raises(ValueError, match="15"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT, min_source_files=1)


def test_rejects_unanchored_invented_source_files_from_diversity_count():
    candidate = case("What does Calculator.add return?")
    candidate["source_files"] = [
        "math_utils.py",
        *(f"invented_{index}.py" for index in range(19)),
    ]

    with pytest.raises(ValueError, match="source file|anchor"):
        validate_case_quality([candidate], source_root=FIXTURE_ROOT)


def test_rejects_source_files_outside_the_source_root():
    source_file = str(Path(__file__).resolve())
    candidate = case(
        "What does Calculator.add return?",
        file=source_file,
        anchor=f"{source_file}:7-9",
    )

    with pytest.raises(ValueError, match="outside source root"):
        validate_case_quality([candidate], source_root=FIXTURE_ROOT)


def test_rejects_synthetic_sliding_anchors():
    cases = [
        case(
            f"What does symbol {chr(ord('a') + index)} do?",
            case_id=f"AQ{index:02d}",
            anchor=f"math_utils.py:{index + 7}-{index + 9}",
        )
        for index in range(4)
    ]

    with pytest.raises(ValueError, match="synthetic|anchor"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_draft_review_without_approved_evidence():
    with pytest.raises(ValueError, match="draft|evidence"):
        validate_review_evidence(
            {"status": "draft", "evidence_grade": "draft"},
            source_validated=True,
        )


def test_rejects_approved_review_without_evidence_grade():
    with pytest.raises(ValueError, match="evidence"):
        validate_review_evidence(
            {
                "status": "approved",
                "reviewer": "Codex",
                "reviewer_type": "ai_source_review",
                "reviewed_at": "2026-08-08",
            },
            source_validated=True,
        )


def test_rejects_approved_ai_review_without_source_validation():
    with pytest.raises(ValueError, match="source validation"):
        validate_review_evidence(
            {
                "status": "approved",
                "evidence_grade": "approved_ai_source_review",
                "reviewer": "Codex",
                "reviewer_type": "ai_source_review",
                "reviewed_at": "2026-08-08",
            },
            source_validated=False,
        )


def test_accepts_approved_ai_review_after_source_validation():
    validate_review_evidence(
        {
            "status": "approved",
            "evidence_grade": "approved_ai_source_review",
            "reviewer": "Codex",
            "reviewer_type": "ai_source_review",
            "reviewed_at": "2026-08-08",
        },
        source_validated=True,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "llama_index_core_100_answer_quality.json",
        "transformers_100_answer_quality.json",
        "langchain_core_100_answer_quality.json",
    ],
)
def test_final_dataset_uses_at_least_20_source_files(filename: str):
    payload = json.loads((GOLD_DIR / filename).read_text())
    source_files = {
        source_file
        for item in payload["cases"]
        for source_file in item["source_files"]
    }

    assert len(source_files) >= 20


def test_preserved_case_helper_accepts_historical_prefix():
    historical_cases = [{"id": "AQ01"}, {"id": "AQ02"}]
    final_cases = [*historical_cases, {"id": "AQ03"}]

    assert_preserved_cases(final_cases, historical_cases)


@pytest.mark.parametrize(
    "filename",
    [
        "llama_index_core_100_answer_quality.json",
        "transformers_100_answer_quality.json",
        "langchain_core_100_answer_quality.json",
    ],
)
def test_final_dataset_has_required_difficulty_distribution(filename: str):
    payload = json.loads((GOLD_DIR / filename).read_text())
    validate_distribution(payload["cases"])
