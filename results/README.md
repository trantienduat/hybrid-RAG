# Evaluation result status

The checked-in `evaluation_results.json` and `evaluation_report.md` are a
historical schema-v2 run from 2026-07-26. They predate the schema-v3 identity
repair and contain no RAGAS answer-quality scores (`ragas` is `null`).

Do not cite them as current evidence. Regenerate both files with
`scripts/run_full_evaluation.py` against a schema-v3 index; the current runner
fails on incomplete ground truth, retrieval errors, provenance/model mismatch,
or an index change during the run.
