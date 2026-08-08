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
