"""Regression gates for historical and regenerated answer-quality datasets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

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


def _load_generator_module():
    script_path = ROOT / "scripts" / "create_100_gold_datasets.py"
    spec = importlib.util.spec_from_file_location("create_100_gold_datasets", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_assemble_dataset_is_deterministic_preserves_prefix_and_copies_cases(tmp_path: Path):
    historical = {
        "name": "historical",
        "schema_version": 1,
        "cases": [{"id": "AQ01", "nested": {"value": "preserved"}}],
    }
    additions = {"cases": [{"id": "AQ02", "nested": {"value": "addition"}}]}
    metadata = {
        "name": "assembled",
        "source_identity": "sha256:" + "a" * 64,
        "source": {"package": "example", "details": {"version": "1.0"}},
    }
    historical_path = tmp_path / "historical.json"
    additions_path = tmp_path / "additions.json"
    historical_path.write_text(json.dumps(historical), encoding="utf-8")
    additions_path.write_text(json.dumps(additions), encoding="utf-8")
    historical_before = json.loads(json.dumps(historical))
    additions_before = json.loads(json.dumps(additions))
    metadata_before = json.loads(json.dumps(metadata))

    generator = _load_generator_module()
    first = generator.assemble_dataset(historical_path, additions_path, metadata)
    second = generator.assemble_dataset(historical_path, additions_path, metadata)

    canonical = lambda payload: json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    assert canonical(first).encode("utf-8") == canonical(second).encode("utf-8")
    assert first["cases"][: len(historical["cases"])] == historical["cases"]
    first["cases"][0]["nested"]["value"] = "changed"
    first["source"]["details"]["version"] = "changed"
    assert historical == historical_before
    assert additions == additions_before
    assert metadata == metadata_before


def _initialize_git_snapshot(tmp_path: Path) -> tuple[Path, str]:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    tracked = snapshot_root / "tracked.py"
    tracked.write_text("VALUE = 'committed'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(snapshot_root)], check=True)
    subprocess.run(["git", "-C", str(snapshot_root), "add", "tracked.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(snapshot_root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "snapshot",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(snapshot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return snapshot_root, commit


def test_git_snapshot_identity_rejects_dirty_tracked_python_source(tmp_path: Path):
    generator = _load_generator_module()
    snapshot_root, commit = _initialize_git_snapshot(tmp_path)
    source_identity = f"git:https://example.test/repo.git@{commit}:."
    (snapshot_root / "tracked.py").write_text("VALUE = 'modified'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean"):
        generator.verify_snapshot_identity(snapshot_root, snapshot_root, source_identity)


def test_git_snapshot_identity_rejects_untracked_python_source(tmp_path: Path):
    generator = _load_generator_module()
    snapshot_root, commit = _initialize_git_snapshot(tmp_path)
    source_identity = f"git:https://example.test/repo.git@{commit}:."
    (snapshot_root / "untracked.py").write_text("VALUE = 'new'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean"):
        generator.verify_snapshot_identity(snapshot_root, snapshot_root, source_identity)


def test_historical_fixture_hash_rejects_modified_known_fixture(tmp_path: Path):
    generator = _load_generator_module()
    filename = "llama_index_core_answer_quality_v1.json"
    fixture = tmp_path / filename
    fixture.write_bytes((GOLD_DIR / filename).read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Historical fixture hash mismatch"):
        generator.verify_historical_fixture(fixture)


def _valid_cases(source_root: Path) -> list[dict]:
    cases = []
    for index in range(100):
        source_file = f"source_{index % 20}.py"
        (source_root / source_file).write_text("documented behavior\n", encoding="utf-8")
        difficulty = "simple" if index < 35 else "medium" if index < 70 else "hard"
        cases.append(
            {
                "id": f"AQ{index:03d}",
                "difficulty": difficulty,
                "question": (
                    f"What does operation {chr(97 + index // 26)}{chr(97 + index % 26)} return?"
                ),
                "reference_answer": "It returns the documented result.",
                "source_files": [source_file],
                "source_anchors": [f"{source_file}:1-1"],
                "reference_contexts": ["The source documents the behavior."],
            }
        )
    return cases


def test_generator_requires_only_the_selected_source_root(tmp_path: Path):
    generator = _load_generator_module()

    args = generator.parse_args(["--only", "llama-index", "--llama-root", str(tmp_path)])

    assert args.only == "llama-index"
    assert args.llama_root == tmp_path
    with pytest.raises(SystemExit):
        generator.parse_args(["--only", "transformers"])


def test_generator_resolves_source_root_inside_snapshot(tmp_path: Path):
    generator = _load_generator_module()
    expected = tmp_path / "llama_index" / "core"
    expected.mkdir(parents=True)

    assert (
        generator.resolve_source_root(tmp_path, generator.DATASETS["llama-index"])
        == expected
    )


def test_approved_generation_rejects_missing_review_evidence_after_source_validation(
    tmp_path: Path,
):
    generator = _load_generator_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    cases = _valid_cases(source_root)
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    output_path = tmp_path / "output.json"
    dataset = {
        "name": "approved",
        "schema_version": 1,
        "source_identity": "sha256:" + "b" * 64,
        "review": {
            "status": "approved",
            "reviewer": "Codex",
            "reviewer_type": "ai_source_review",
            "reviewed_at": "2026-08-08",
        },
        "cases": cases,
    }

    with pytest.raises(ValueError, match="evidence grade"):
        generator.validate_and_write_dataset(
            dataset,
            historical_path=historical_path,
            output_path=output_path,
            source_root=source_root,
        )

    assert not output_path.exists()


def test_loader_failure_keeps_existing_output_unchanged(tmp_path: Path):
    generator = _load_generator_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    cases = _valid_cases(source_root)
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    output_path = tmp_path / "output.json"
    original = b"existing output\n"
    output_path.write_bytes(original)
    dataset = {
        "name": "invalid-loader-schema",
        "schema_version": 1,
        "source_identity": "not-an-immutable-identity",
        "review": {"status": "draft"},
        "cases": cases,
    }

    with pytest.raises(ValueError, match="immutable source_identity"):
        generator.validate_and_write_dataset(
            dataset,
            historical_path=historical_path,
            output_path=output_path,
            source_root=source_root,
        )

    assert output_path.read_bytes() == original


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
