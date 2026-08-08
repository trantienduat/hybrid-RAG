"""Regression gates for historical and regenerated answer-quality datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hybrid_rag.eval.gold_validation import validate_case_quality


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
    anchor: str = "module.py:1-3",
    file: str = "module.py",
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
    cases = [
        case("How does the component work?", case_id="AQ01"),
        case("How does the module work?", case_id="AQ02"),
    ]

    with pytest.raises(ValueError, match="normalized question"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_duplicate_normalized_questions():
    cases = [
        case("What does Widget.run return?", case_id="AQ01"),
        case(" what does widget.run return ", case_id="AQ02"),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_invalid_anchor_bounds():
    candidate = case("What does Widget.run return?", anchor="module.py:10-99")

    with pytest.raises(ValueError, match="anchor"):
        validate_case_quality([candidate], source_root=FIXTURE_ROOT)


def test_rejects_source_file_concentration():
    cases = [
        case(f"What does symbol_{i} do?", case_id=f"AQ{i:02d}", file="module.py")
        for i in range(16)
    ]

    with pytest.raises(ValueError, match="15"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT, min_source_files=1)


def test_rejects_synthetic_sliding_anchors():
    cases = [
        case(
            f"What does symbol_{i} do?",
            case_id=f"AQ{i:02d}",
            anchor=f"math_utils.py:{i + 1}-{i + 3}",
            file="math_utils.py",
        )
        for i in range(4)
    ]

    with pytest.raises(ValueError, match="synthetic|anchor"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


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


def test_resolves_anchors_against_pinned_snapshot(tmp_path):
    pinned_root = tmp_path / "snapshot-v1"
    working_root = tmp_path / "working-tree"
    pinned_root.mkdir()
    working_root.mkdir()
    pinned_lines = ["# pinned source\n"] * 20
    pinned_lines[9:12] = ["def run():\n", "    return 1\n", "\n"]
    (pinned_root / "module.py").write_text("".join(pinned_lines), encoding="utf-8")
    (working_root / "module.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    candidate = case("What does Widget.run return?", anchor="module.py:10-12")

    validate_case_quality(
        [candidate],
        source_root=working_root,
        pinned_snapshot=pinned_root,
    )


def test_rejects_draft_review_without_approved_evidence():
    with pytest.raises(ValueError, match="draft|evidence"):
        validate_case_quality(
            [case("What does Widget.run return?")],
            source_root=FIXTURE_ROOT,
            review={"status": "draft", "evidence_grade": "draft"},
        )


def test_rejects_approved_review_without_source_evidence():
    with pytest.raises(ValueError, match="evidence"):
        validate_case_quality(
            [case("What does Widget.run return?")],
            source_root=FIXTURE_ROOT,
            review={
                "status": "approved",
                "reviewer": "Codex",
                "reviewer_type": "ai_source_review",
                "reviewed_at": "2026-08-08",
            },
        )


@pytest.mark.parametrize(
    ("final_name", "historical_name", "preserved_count"),
    [
        (
            "llama_index_core_100_answer_quality.json",
            "llama_index_core_answer_quality_v1.json",
            30,
        ),
        (
            "transformers_100_answer_quality.json",
            "transformers_v5_9_answer_quality_v1.json",
            10,
        ),
        (
            "langchain_core_100_answer_quality.json",
            "langchain_core_v1_4_7_answer_quality_v1.json",
            10,
        ),
    ],
)
def test_final_dataset_preserves_historical_cases(
    final_name: str, historical_name: str, preserved_count: int
):
    final = json.loads((GOLD_DIR / final_name).read_text())
    historical = json.loads((GOLD_DIR / historical_name).read_text())

    assert final["cases"][:preserved_count] == historical["cases"]


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
    assert len(payload["cases"]) == 100
    assert {
        difficulty: sum(case["difficulty"] == difficulty for case in payload["cases"])
        for difficulty in ("simple", "medium", "hard")
    } == {"simple": 35, "medium": 35, "hard": 30}
