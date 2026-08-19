# Evaluation result status

The checked-in `evaluation_results.json` and `evaluation_report.md` are a
historical schema-v2 run from 2026-07-26. They predate the schema-v3 identity
repair and contain no RAGAS answer-quality scores (`ragas` is `null`).

Do not cite them as current evidence. Regenerate both files with
`scripts/run_full_evaluation.py` against a schema-v3 index; the current runner
fails on incomplete ground truth, retrieval errors, provenance/model mismatch,
or an index change during the run.

The current generated-answer evidence is
`answer_quality_benchmark_schema3.json`. It contains 180 complete records for
30 reviewed LlamaIndex Core questions, two retrieval modes, and three repeats.
Reproduce the case-level uncertainty analysis with:

```bash
python3 scripts/analyze_answer_quality_statistics.py
```

The command writes `answer_quality_benchmark_statistics.json`. It first
averages repeats within each question, then performs a deterministic paired
nonparametric bootstrap over the 30 question pairs (10,000 resamples, seed 42).
This avoids treating repeated generations of the same question as independent
questions. The resulting confidence intervals describe this pinned workload;
they do not establish cross-repository or cross-language generality.
