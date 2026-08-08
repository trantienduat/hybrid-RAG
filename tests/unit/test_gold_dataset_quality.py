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
    anchor: str = "module.py:1-3",
    file: str = "module.py",
) -> dict:
    return {
        "id": "AQ01",
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
    cases = [case("How does component #11 work?"), case("How does component #12 work?")]

    with pytest.raises(ValueError, match="normalized question"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT)


def test_rejects_invalid_anchor_bounds():
    candidate = case("What does Widget.run return?", anchor="module.py:10-99")

    with pytest.raises(ValueError, match="anchor"):
        validate_case_quality([candidate], source_root=FIXTURE_ROOT)


def test_rejects_source_file_concentration():
    cases = [case(f"What does symbol_{i} do?", file="module.py") for i in range(16)]

    with pytest.raises(ValueError, match="15"):
        validate_case_quality(cases, source_root=FIXTURE_ROOT, min_source_files=1)


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
    assert {
        difficulty: sum(case["difficulty"] == difficulty for case in payload["cases"])
        for difficulty in ("simple", "medium", "hard")
    } == {"simple": 35, "medium": 35, "hard": 30}
