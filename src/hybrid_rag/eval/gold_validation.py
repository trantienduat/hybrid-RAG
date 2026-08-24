"""Reusable quality checks for curated gold answer datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_NUMBERED_COMPONENT = re.compile(r"\bcomponent\s*#?\d+\b", re.IGNORECASE)
_GENERIC_COMPONENT_NAME = re.compile(
    r"\b(?:the\s+)?(?:component|module)\s+"
    r"(?:work|works|do|does|behave|behaves|operate|operates|function|functions|"
    r"handle|handles|return|returns)\b",
    re.IGNORECASE,
)
_ANCHOR = re.compile(r"(.+):(\d+)-(\d+)")


def normalize_question(question: str) -> str:
    """Normalize templates and whitespace before checking question uniqueness."""
    normalized = _NUMBERED_COMPONENT.sub("component #N", question)
    normalized = re.sub(r"\b\d+\b", "N", normalized.lower())
    return " ".join(normalized.split())


def _source_file_path(source_root: Path, source_file: str) -> Path:
    path = source_root / source_file
    try:
        path.resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError(f"source file is outside source root: {source_file}") from exc
    return path


def resolve_anchor(source_root: Path, anchor: str) -> tuple[Path, int, int, str]:
    """Return the file, inclusive bounds, and non-empty excerpt for an anchor."""
    match = _ANCHOR.fullmatch(anchor)
    if match is None:
        raise ValueError(f"Invalid source anchor: {anchor}")
    relative, start_text, end_text = match.groups()
    path = _source_file_path(source_root, relative)
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


def _reject_synthetic_sliding_anchors(
    anchors_by_file: Mapping[str, list[tuple[int, int, str]]],
) -> None:
    for source_file, anchors in anchors_by_file.items():
        runs_by_width: dict[int, tuple[int, int, set[str]]] = {}
        for start, end, case_id in sorted(anchors):
            width = end - start
            previous = runs_by_width.get(width)
            if previous is not None and start == previous[0] + 1:
                run_length = previous[1] + 1
                case_ids = previous[2] | {case_id}
            else:
                run_length = 1
                case_ids = {case_id}
            runs_by_width[width] = (start, run_length, case_ids)
            if run_length >= 3 and len(case_ids) >= 3:
                raise ValueError(
                    f"synthetic sliding source anchors detected for {source_file}"
                )


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
    anchored_files: set[str] = set()
    anchors_by_file: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for case in cases:
        case_id = case["id"]
        if case_id in seen_ids:
            raise ValueError(f"Duplicate gold case ID: {case_id}")
        seen_ids.add(case_id)

        question = case["question"]
        if _NUMBERED_COMPONENT.search(question):
            raise ValueError(f"Banned normalized question template: {question}")
        if _GENERIC_COMPONENT_NAME.search(question):
            raise ValueError(f"Banned generic normalized question template: {question}")
        normalized_question = normalize_question(question)
        if normalized_question in seen_questions:
            raise ValueError(f"duplicate normalized question: {normalized_question}")
        seen_questions.add(normalized_question)

        source_files = case["source_files"]
        for source_file in source_files:
            if not _source_file_path(source_root, source_file).is_file():
                raise ValueError(f"source file does not exist: {source_file}")
            file_counts[source_file] += 1

        for anchor in case["source_anchors"]:
            match = _ANCHOR.fullmatch(anchor)
            if match is None or match.group(1) not in source_files:
                raise ValueError(f"Source anchor is not declared by its case: {anchor}")
            resolve_anchor(source_root, anchor)
            relative, start_text, end_text = match.groups()
            anchored_files.add(relative)
            anchors_by_file[relative].append((int(start_text), int(end_text), case_id))

    unanchored_files = set(file_counts) - anchored_files
    if unanchored_files:
        raise ValueError(
            "source file has no declared anchor: "
            f"{', '.join(sorted(unanchored_files))}"
        )

    _reject_synthetic_sliding_anchors(anchors_by_file)

    verified_file_counts = Counter(
        {
            source_file: count
            for source_file, count in file_counts.items()
            if source_file in anchored_files
        }
    )
    overrepresented = [
        source_file
        for source_file, count in verified_file_counts.items()
        if count > max_cases_per_file
    ]
    if overrepresented:
        raise ValueError(
            f"No source file may appear in more than {max_cases_per_file} cases: "
            f"{', '.join(overrepresented)}"
        )
    if len(verified_file_counts) < min_source_files:
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


def validate_review_evidence(
    review: Mapping[str, Any],
    *,
    source_validated: bool,
) -> None:
    """Require provenance evidence before an approved review can be consumed."""
    status = str(review.get("status", "")).strip()
    reviewer = str(review.get("reviewer", "")).strip()
    reviewer_type = str(review.get("reviewer_type", "")).strip()
    reviewed_at = str(review.get("reviewed_at", "")).strip()
    evidence_grade = str(review.get("evidence_grade", "")).strip()

    if status != "approved" or not reviewer or not reviewed_at:
        raise ValueError("review evidence must be approved and attributable")
    expected_grades = {
        "human": "approved_human_review",
        "ai_source_review": "approved_ai_source_review",
    }
    if evidence_grade != expected_grades.get(reviewer_type):
        raise ValueError("review evidence grade does not match reviewer provenance")
    if reviewer_type == "ai_source_review" and not source_validated:
        raise ValueError("approved_ai_source_review requires source validation")
