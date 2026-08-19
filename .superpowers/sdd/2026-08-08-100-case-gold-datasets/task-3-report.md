# Task 3 Report: Deterministic Gold-Case Assembler

## Delivered

- Replaced generated-question loops with `assemble_dataset`, which reads the immutable historical dataset and a curated additions catalog, deep-copies cases, and emits canonical schema-version-1 data.
- Added empty, schema-valid additions catalogs for LlamaIndex, Transformers, and LangChain Core.
- Added a generator CLI with `--only` and per-repository snapshot roots. A selected dataset requires only its own root; generating all datasets requires all three roots.
- Resolves snapshot roots to the exact source subdirectory used by anchors and verifies the pinned source identity before validation.
- Added pre-write validation for the preserved historical prefix, exact distribution, case quality/source anchors, and approved review evidence. Approved output reaches `validate_review_evidence` only after source-case validation succeeds.
- Canonical writing uses two-space JSON, UTF-8, and one trailing newline, then reloads through `load_gold_dataset`.

## TDD Evidence

1. `test_assemble_dataset_is_deterministic_preserves_prefix_and_copies_cases` initially failed because `assemble_dataset` was absent, then passed after the copy-based assembler was added.
2. `test_generator_requires_only_the_selected_source_root` initially failed because `parse_args` was absent, then passed after CLI parsing was added.
3. `test_generator_resolves_source_root_inside_snapshot` failed before snapshot-root resolution was implemented, then passed.
4. `test_approved_generation_rejects_missing_review_evidence_after_source_validation` initially failed when an approved dataset without an evidence grade was written; it passed after the review-evidence gate was restored before write.

## Verification

- `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py -k 'assemble or deterministic or generator or approved_generation'` — 4 passed.
- `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py` — 22 passed; 2 expected placeholder failures remain.
- `PYTHONPATH=src .venv/bin/python scripts/create_100_gold_datasets.py --llama-root /private/tmp/hybrid-rag-gold-sources/llama-index-core-0.14.21 --only llama-index --review-status approved` — snapshot identity and source-root resolution passed; the intentionally empty catalog then stopped at the required 100-case distribution gate before writing.
- `git diff --check` — no whitespace errors.

## Fix Round 2

- Replaced porcelain-text parsing for untracked source detection with NUL-delimited `git ls-files --others --exclude-standard -z`, scoped to the selected source root. This preserves rejection of ordinary untracked Python files and handles Git-quoted paths containing spaces.
- Added `test_git_snapshot_identity_rejects_quoted_untracked_python_source` using the untracked filename `untracked source.py`.

### Fix-Round-2 Verification

- RED: the new quoted-path regression test failed before the parser fix because no `ValueError` was raised.
- GREEN: `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py -k 'snapshot_identity'` — 3 passed.
- `git diff --check` — no whitespace errors.

## Known Concerns

- The current Transformers and LangChain Core 100-case outputs remain invalid against the minimum-20-source-file diversity gate (6 and 5 files respectively). These are the expected Task 5 and Task 6 placeholders.
- All additions catalogs are intentionally empty until Tasks 4 through 6; consequently, the assembler correctly refuses to regenerate a 100-case production output today.

## Scope

No historical or existing generated dataset file changed. This task is limited to the assembler, its tests, empty catalogs, and this report.

Commit: `eb0d993 refactor(eval): assemble curated gold cases`.

## Fix Round

- Git-backed snapshots now reject any tracked modification and untracked Python files under the selected source root, after confirming the pinned commit.
- Production historical fixtures are SHA-256 checked against all three immutable pinned hashes before assembly.
- `assemble_dataset` deep-copies metadata as well as cases.
- Output is now rendered to a same-directory temporary file, schema-loaded there, and atomically replaced only after that succeeds; a loader failure leaves an existing output intact.

### Fix-Round Verification

- RED: `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py -k 'assemble or snapshot_identity or historical_fixture_hash or loader_failure'` — 5 failed before the fixes (metadata aliasing, dirty tracked and untracked Git sources accepted, absent fixture hash verification, and output overwrite on loader failure).
- GREEN: same focused command — 5 passed.
- `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py` — 26 passed; 2 expected placeholder diversity failures remain for Transformers (6/20) and LangChain Core (5/20).
- `git diff --check` — no whitespace errors.
