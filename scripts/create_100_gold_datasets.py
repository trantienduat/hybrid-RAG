#!/usr/bin/env python3
"""Assemble and validate curated answer-quality gold datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.eval.answer_quality import load_gold_dataset
from hybrid_rag.eval.gold_validation import (
    assert_preserved_cases,
    validate_case_quality,
    validate_distribution,
    validate_review_evidence,
)


GOLD_DIR = ROOT / "eval" / "gold"
CATALOG_DIR = GOLD_DIR / "catalogs"

HISTORICAL_FIXTURE_SHA256 = {
    "llama_index_core_answer_quality_v1.json": (
        "c11a4d3fb7dd9645e81a8e060a5c0a719fd3b005b81c9827677717918f03ec37"
    ),
    "transformers_v5_9_answer_quality_v1.json": (
        "066914fed173337ee78c1f644348c252091275101faad2d8b414bb15fcf4c894"
    ),
    "langchain_core_v1_4_7_answer_quality_v1.json": (
        "8f6e61fd2bf1f22bfd8adaa5513283d35a4a68ca2f2df382e10de1cf8b718d95"
    ),
}

SUPPLEMENTARY_V2_ANCHOR_CORRECTIONS = {
    "sha256:13b55bcf111885b7c2c9c07886b5656769041331c81c28bcf3cbfb87939ec509": {
        "AQ08": {
            "response_synthesizers/factory.py:33-170": (
                "response_synthesizers/factory.py:33-151"
            ),
        },
        "AQ23": {
            "retrievers/fusion_retriever.py:232-320": (
                "retrievers/fusion_retriever.py:232-317"
            ),
        },
    },
}

DATASETS: dict[str, dict[str, Any]] = {
    "llama-index": {
        "source_subdir": Path("llama_index/core"),
        "historical": GOLD_DIR / "llama_index_core_answer_quality_v1.json",
        "additions": CATALOG_DIR / "llama_index_core_additions.json",
        "output": GOLD_DIR / "llama_index_core_100_answer_quality.json",
        "metadata": {
            "name": "llama-index-core-100-answer-quality-v2",
            "source_identity": "sha256:13b55bcf111885b7c2c9c07886b5656769041331c81c28bcf3cbfb87939ec509",
            "source": {
                "package": "llama-index-core",
                "version": "0.14.21",
                "artifact_sha256": "4a807d31e54d066068e076eb4d066efbf95e2d2a00dcbe0eba3d9340a04cad42",
            },
        },
    },
    "transformers": {
        "source_subdir": Path("."),
        "historical": GOLD_DIR / "transformers_v5_9_answer_quality_v1.json",
        "additions": CATALOG_DIR / "transformers_additions.json",
        "output": GOLD_DIR / "transformers_100_answer_quality.json",
        "metadata": {
            "name": "transformers-100-answer-quality-v2",
            "source_identity": "git:https://github.com/huggingface/transformers.git@0a2757da521a7a49b8143d9e0c938f08747d682e:.",
        },
    },
    "langchain-core": {
        "source_subdir": Path("libs/core"),
        "historical": GOLD_DIR / "langchain_core_v1_4_7_answer_quality_v1.json",
        "additions": CATALOG_DIR / "langchain_core_additions.json",
        "output": GOLD_DIR / "langchain_core_100_answer_quality.json",
        "metadata": {
            "name": "langchain-core-100-answer-quality-v2",
            "source_identity": "git:https://github.com/langchain-ai/langchain.git@51578289bb1f696a643e0740be1441039d8af8ce:libs/core",
        },
    },
}


def assemble_dataset(
    historical_path: Path,
    additions_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble immutable historical cases with curated additions."""
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    additions = json.loads(additions_path.read_text(encoding="utf-8"))
    cases = [copy.deepcopy(case) for case in historical["cases"]]
    _apply_supplementary_v2_anchor_corrections(
        cases, str(metadata.get("source_identity", ""))
    )
    cases.extend(copy.deepcopy(case) for case in additions["cases"])
    assembled = copy.deepcopy(dict(metadata))
    assembled.update({"schema_version": 1, "cases": cases})
    return assembled


def _apply_supplementary_v2_anchor_corrections(
    cases: Sequence[dict[str, Any]], source_identity: str
) -> None:
    """Correct audited over-broad historical anchors in supplementary output."""
    corrections = SUPPLEMENTARY_V2_ANCHOR_CORRECTIONS.get(source_identity, {})
    if not corrections:
        return

    cases_by_id = {case["id"]: case for case in cases}
    for case_id, anchor_replacements in corrections.items():
        if case_id not in cases_by_id:
            raise ValueError(f"Missing historical case required for correction: {case_id}")
        anchors = cases_by_id[case_id]["source_anchors"]
        for original, corrected in anchor_replacements.items():
            if anchors.count(original) != 1:
                raise ValueError(
                    f"{case_id} expected historical source anchor exactly once: "
                    f"{original}"
                )
            anchors[anchors.index(original)] = corrected


def verify_historical_fixture(historical_path: Path) -> None:
    """Require a production historical fixture to match its pinned bytes."""
    expected = HISTORICAL_FIXTURE_SHA256.get(historical_path.name)
    if expected is None:
        raise ValueError(f"Unknown historical fixture: {historical_path.name}")
    actual = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Historical fixture hash mismatch: expected {expected}, got {actual}"
        )


def _review_metadata(review_status: str) -> dict[str, str]:
    review = {"status": review_status}
    if review_status == "approved":
        review.update(
            {
                "reviewer": "Codex",
                "reviewer_type": "ai_source_review",
                "reviewed_at": "2026-08-08",
                "evidence_grade": "approved_ai_source_review",
            }
        )
    return review


def resolve_source_root(snapshot_root: Path, spec: Mapping[str, Any]) -> Path:
    """Resolve a repository snapshot to the directory named by its anchors."""
    source_root = snapshot_root / spec["source_subdir"]
    if not source_root.is_dir():
        raise ValueError(f"Source root does not exist: {source_root}")
    return source_root


def _directory_python_digest(source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_snapshot_identity(
    snapshot_root: Path,
    source_root: Path,
    source_identity: str,
) -> None:
    """Require the supplied immutable snapshot to match its dataset identity."""
    if source_identity.startswith("sha256:"):
        actual = _directory_python_digest(source_root)
        expected = source_identity.removeprefix("sha256:")
        if actual != expected:
            raise ValueError(f"Snapshot identity mismatch: expected {expected}, got {actual}")
        return

    expected = source_identity.split("@", maxsplit=1)[1].split(":", maxsplit=1)[0]
    result = subprocess.run(
        ["git", "-C", str(snapshot_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    actual = result.stdout.strip()
    if result.returncode or actual != expected:
        raise ValueError(f"Snapshot identity mismatch: expected {expected}, got {actual}")

    tracked = subprocess.run(
        ["git", "-C", str(snapshot_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        check=False,
        text=True,
    )
    if tracked.returncode or tracked.stdout.strip():
        raise ValueError("Git snapshot must be clean: tracked modifications found")

    source_relative = source_root.relative_to(snapshot_root)
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(snapshot_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "." if source_relative == Path(".") else source_relative.as_posix(),
        ],
        capture_output=True,
        check=False,
    )
    if untracked.returncode:
        raise ValueError("Git snapshot must be clean: unable to inspect untracked files")
    for raw_path in untracked.stdout.split(b"\0"):
        if not raw_path:
            continue
        candidate = Path(os.fsdecode(raw_path))
        if candidate.suffix == ".py":
            raise ValueError("Git snapshot must be clean: untracked Python source found")


def validate_and_write_dataset(
    dataset: Mapping[str, Any],
    *,
    historical_path: Path,
    output_path: Path,
    source_root: Path,
) -> None:
    """Validate a complete dataset, then write canonical JSON and reload it."""
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    expected_historical_cases = [
        copy.deepcopy(case) for case in historical["cases"]
    ]
    _apply_supplementary_v2_anchor_corrections(
        expected_historical_cases, str(dataset.get("source_identity", ""))
    )
    cases = dataset["cases"]
    assert_preserved_cases(cases, expected_historical_cases)
    validate_distribution(cases)
    validate_case_quality(cases, source_root=source_root)

    review = dataset["review"]
    if review["status"] == "approved":
        validate_review_evidence(review, source_validated=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    try:
        load_gold_dataset(temporary, allow_draft=review["status"] == "draft")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-root", type=Path)
    parser.add_argument("--transformers-root", type=Path)
    parser.add_argument("--langchain-root", type=Path)
    parser.add_argument("--only", choices=tuple(DATASETS))
    parser.add_argument("--review-status", choices=("draft", "approved"), default="draft")
    args = parser.parse_args(argv)

    required_roots = {
        "llama-index": "llama_root",
        "transformers": "transformers_root",
        "langchain-core": "langchain_root",
    }
    selected = (args.only,) if args.only else tuple(DATASETS)
    for dataset_name in selected:
        root_name = required_roots[dataset_name]
        if getattr(args, root_name) is None:
            parser.error(f"--{root_name.replace('_', '-')} is required for {dataset_name}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    roots = {
        "llama-index": args.llama_root,
        "transformers": args.transformers_root,
        "langchain-core": args.langchain_root,
    }
    selected = (args.only,) if args.only else tuple(DATASETS)

    for dataset_name in selected:
        spec = DATASETS[dataset_name]
        verify_historical_fixture(spec["historical"])
        metadata = {**spec["metadata"], "review": _review_metadata(args.review_status)}
        dataset = assemble_dataset(spec["historical"], spec["additions"], metadata)
        source_root = resolve_source_root(roots[dataset_name], spec)
        verify_snapshot_identity(
            roots[dataset_name],
            source_root,
            metadata["source_identity"],
        )
        validate_and_write_dataset(
            dataset,
            historical_path=spec["historical"],
            output_path=spec["output"],
            source_root=source_root,
        )
        print(f"Wrote and validated {spec['output'].name} ({len(dataset['cases'])} cases)")


if __name__ == "__main__":
    main()
