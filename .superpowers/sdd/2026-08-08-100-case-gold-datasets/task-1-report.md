# Task 1 Report: Lock Historical Cases and Define Quality Failures

## Implementation summary

Added `tests/unit/test_gold_dataset_quality.py` with regression expectations for:

- SHA-256 immutability of the three historical gold datasets.
- Requiring the regenerated datasets to preserve the specified historical-case prefixes.
- The required 35 simple / 35 medium / 30 hard difficulty distribution in each regenerated dataset.
- Rejection of numbered-component question templates.
- Rejection of out-of-bounds source anchors.
- Rejection of source-file concentration above 15 cases when at least one source file is required.

No production validation code was added. This task intentionally leaves the validator import and regenerated quality gates RED for the next implementation task.

## Commands and test results

| Command | Result |
| --- | --- |
| `shasum -a 256 eval/gold/*_answer_quality_v1.json` | PASS; all three values match the brief exactly. |
| `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py` | EXPECTED RED; pytest exits 2 during collection because `hybrid_rag.eval.gold_validation` does not exist yet. |
| `.venv/bin/python -m py_compile tests/unit/test_gold_dataset_quality.py` | PASS. |
| `git diff --check` | PASS. |

## TDD RED evidence and why expected

The focused pytest run failed during test collection with:

```text
E   ModuleNotFoundError: No module named 'hybrid_rag.eval.gold_validation'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.05s
```

This is the expected RED state specified by the brief: the tests define the validator contract before `hybrid_rag.eval.gold_validation` is implemented. The failure is caused by the deliberately missing production module, not by a test assertion or syntax error.

## Files changed

- `tests/unit/test_gold_dataset_quality.py`
- `.superpowers/sdd/2026-08-08-100-case-gold-datasets/task-1-report.md`

## Self-review

- Confirmed all three historical SHA-256 values against the checked-in files before adding the tests.
- Confirmed the preservation counts are 30, 10, and 10 as specified.
- Confirmed the test module contains no production implementation or dataset edits.
- Confirmed syntax compilation and `git diff --check` succeed.
- Confirmed the focused test fails for the intended missing-validator reason.

## Concerns

- The focused test cannot execute individual assertions until Task 2 adds `hybrid_rag.eval.gold_validation`; the current collection error is intentional.
- The checked-in regenerated datasets currently contain quality patterns targeted by these gates, so the later validator/regeneration work must resolve them before this suite can turn GREEN.

## Fix Round 1

Addressed every review finding without adding production validation:

- Added a generic component/module-name template rejection gate.
- Added duplicate normalized-question rejection using case and whitespace variation.
- Added synthetic sliding-anchor rejection.
- Added a regenerated-dataset gate requiring at least 20 distinct source files.
- Added pinned-snapshot anchor-resolution coverage using a valid anchor in the pinned source and different working-tree contents.
- Added draft-review-without-approved-evidence and approved-review-without-source-evidence gates.
- Added `len(cases) == 100` to the regenerated difficulty-distribution gate.
- Updated `case()` to accept `case_id`; concentration and synthetic-anchor cases now use unique IDs.

### Fix Round 1 focused test evidence

Command:

```text
PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py
```

Expected RED result:

```text
E   ModuleNotFoundError: No module named 'hybrid_rag.eval.gold_validation'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.06s
PYTEST_EXIT=2
```

The test names covering the requested gates are:

```text
test_rejects_numbered_component_template
test_rejects_generic_component_name_template
test_rejects_duplicate_normalized_questions
test_rejects_invalid_anchor_bounds
test_rejects_source_file_concentration
test_rejects_synthetic_sliding_anchors
test_final_dataset_uses_at_least_20_source_files
test_resolves_anchors_against_pinned_snapshot
test_rejects_draft_review_without_approved_evidence
test_rejects_approved_review_without_source_evidence
test_final_dataset_has_required_difficulty_distribution
```

Additional checks in the same focused module continue to cover historical hashes and preserved-case equality. `.venv/bin/python -m py_compile tests/unit/test_gold_dataset_quality.py` and `git diff --check` both pass.

### Fix Round 1 concerns

The collection failure is still intentional until Task 2 adds `hybrid_rag.eval.gold_validation`; individual gates cannot execute before that production module exists.
