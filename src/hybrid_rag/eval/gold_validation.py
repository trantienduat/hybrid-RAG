"""Reusable quality checks for curated gold answer datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_NUMBERED_COMPONENT = re.compile(r"\bcomponent\s*#?\d+\b", re.IGNORECASE)
_ANCHOR = re.compile(r"(.+):(\d+)-(\d+)")


def normalize_question(question: str) -> str:
    """Normalize templates and whitespace before checking question uniqueness."""
    normalized = _NUMBERED_COMPONENT.sub("component #N", question)
    normalized = re.sub(r"\b\d+\b", "N", normalized.lower())
    return " ".join(normalized.split())


def resolve_anchor(source_root: Path, anchor: str) -> tuple[Path, int, int, str]:
    """Return the file, inclusive bounds, and non-empty excerpt for an anchor."""
    match = _ANCHOR.fullmatch(anchor)
    if match is None:
        raise ValueError(f"Invalid source anchor: {anchor}")
    relative, start_text, end_text = match.groups()
    path = source_root / relative
    if not path.is_file():
        raise ValueError(f"Source anchor is outside file bounds: {anchor}")
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = int(start_text), int(end_text)
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"Source anchor is outside file bounds: {anchor}")
    excerpt = "\n".join(lines[start - 1 : end]).strip()
    if not excerpt:
        raise ValueError(f"Source anchor is empty: {anchor}")
    return path, start, end, excerpt


def validate_case_quality(
    cases: Sequence[Mapping[str, Any]],
    source_root: Path,
    *,
    max_cases_per_file: int = 15,
    min_source_files: int = 20,
) -> None:
    """Validate reusable structural and source-grounding quality gates."""
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    file_counts: Counter[str] = Counter()

    for case in cases:
        case_id = case["id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate gold case ID: {case_id}")
        seen_ids.add(case_id)

        question = case["question"]
        if _NUMBERED_COMPONENT.search(question):
            raise ValueError(f"Banned normalized question template: {question}")
        normalized_question = normalize_question(question)
        if normalized_question in seen_questions:
            raise ValueError(f"duplicate normalized question: {normalized_question}")
        seen_questions.add(normalized_question)

        source_files = case["source_files"]
        for source_file in source_files:
            file_counts[source_file] += 1

        for anchor in case["source_anchors"]:
            match = _ANCHOR.fullmatch(anchor)
            if match is None or match.group(1) not in source_files:
                raise ValueError(f"Source anchor is not declared by its case: {anchor}")
            resolve_anchor(source_root, anchor)

    overrepresented = [
        source_file
        for source_file, count in file_counts.items()
        if count > max_cases_per_file
    ]
    if overrepresented:
        raise ValueError(
            f"No source file may appear in more than {max_cases_per_file} cases: "
            f"{', '.join(overrepresented)}"
        )
    if len(file_counts) < min_source_files:
        raise ValueError(f"Gold dataset requires at least {min_source_files} source files")


def assert_preserved_cases(
    final_cases: Sequence[Mapping[str, Any]],
    historical_cases: Sequence[Mapping[str, Any]],
) -> None:
    """Require historical cases to remain the exact prefix of final cases."""
    if final_cases[: len(historical_cases)] != historical_cases:
        raise ValueError("Final cases must preserve the historical case prefix")


def validate_distribution(cases: Sequence[Mapping[str, Any]]) -> None:
    """Require the fixed 100-case difficulty distribution."""
    if len(cases) != 100:
        raise ValueError("Gold dataset must contain exactly 100 cases")
    distribution = Counter(case["difficulty"] for case in cases)
    if distribution != {"simple": 35, "medium": 35, "hard": 30}:
        raise ValueError("Gold dataset has an invalid difficulty distribution")
