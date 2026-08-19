# Source-Reviewed 100-Case Gold Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three generated 100-case gold datasets with source-backed cases while preserving every historical thesis and v1 case exactly.

**Architecture:** Keep historical datasets immutable, store only new curated cases in three focused catalog files, and assemble final 100-case datasets through one deterministic generator. A reusable validation module enforces preservation, distributions, semantic uniqueness, source diversity, anchor bounds, and snapshot identity before output can be marked approved.

**Tech Stack:** Python 3.12, dataclasses, pathlib, hashlib, subprocess, JSON, pytest, existing `hybrid_rag.eval.answer_quality` loader.

## Global Constraints

- `eval/gold/llama_index_core_answer_quality_v1.json` remains byte-for-byte unchanged.
- Supplementary v2 corrects only the over-broad AQ08 and AQ23 anchor end lines
  documented in the design; every other historical case field remains exact.
- Historical Transformers and LangChain ten-case datasets remain byte-for-byte unchanged.
- Final datasets contain exactly 100 cases with `35 simple / 35 medium / 30 hard`.
- LlamaIndex reuses 30 historical cases and adds 70 new cases.
- Transformers and LangChain each reuse ten historical cases and add 90 new cases.
- No numbered placeholders, generic component names, synthetic sliding anchors, or duplicate normalized questions.
- Every anchor must resolve inside the pinned source snapshot.
- No source file contributes more than 15 cases; each dataset represents at least 20 source files.
- Output remains `draft` until source and quality validation pass, then becomes `approved_ai_source_review`.
- The new datasets are supplementary cross-repository evidence, not confirmatory thesis evidence.

---

### Task 1: Lock Historical Cases and Define Quality Failures

**Files:**
- Create: `tests/unit/test_gold_dataset_quality.py`
- Read: `eval/gold/llama_index_core_answer_quality_v1.json`
- Read: `eval/gold/transformers_v5_9_answer_quality_v1.json`
- Read: `eval/gold/langchain_core_v1_4_7_answer_quality_v1.json`

**Interfaces:**
- Consumes: existing historical JSON datasets.
- Produces: regression expectations for immutable bytes, preserved case equality, distributions, template rejection, file concentration, and anchor bounds.

- [ ] **Step 1: Record historical file hashes in the test**

```python
HISTORICAL_SHA256 = {
    "llama_index_core_answer_quality_v1.json": "c11a4d3fb7dd9645e81a8e060a5c0a719fd3b005b81c9827677717918f03ec37",
    "transformers_v5_9_answer_quality_v1.json": "066914fed173337ee78c1f644348c252091275101faad2d8b414bb15fcf4c894",
    "langchain_core_v1_4_7_answer_quality_v1.json": "8f6e61fd2bf1f22bfd8adaa5513283d35a4a68ca2f2df382e10de1cf8b718d95",
}

def test_historical_gold_files_are_unchanged():
    for filename, expected in HISTORICAL_SHA256.items():
        payload = (GOLD_DIR / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
```

Reconfirm these values with
`shasum -a 256 eval/gold/*_answer_quality_v1.json` before committing the test.

- [ ] **Step 2: Add tests describing invalid generated datasets**

```python
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
```

- [ ] **Step 3: Add assembly expectations**

```python
@pytest.mark.parametrize(
    ("final_name", "historical_name", "preserved_count"),
    [
        ("llama_index_core_100_answer_quality.json", "llama_index_core_answer_quality_v1.json", 30),
        ("transformers_100_answer_quality.json", "transformers_v5_9_answer_quality_v1.json", 10),
        ("langchain_core_100_answer_quality.json", "langchain_core_v1_4_7_answer_quality_v1.json", 10),
    ],
)
def test_final_dataset_preserves_historical_cases(final_name, historical_name, preserved_count):
    final = json.loads((GOLD_DIR / final_name).read_text())
    historical = json.loads((GOLD_DIR / historical_name).read_text())
    assert final["cases"][:preserved_count] == historical["cases"]
```

- [ ] **Step 4: Run the new tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py`

Expected: FAIL because `hybrid_rag.eval.gold_validation` and valid regenerated datasets do not exist yet.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/unit/test_gold_dataset_quality.py
git commit -m "test(eval): define gold dataset quality gates"
```

### Task 2: Implement Reusable Gold Validation

**Files:**
- Create: `src/hybrid_rag/eval/gold_validation.py`
- Modify: `tests/unit/test_gold_dataset_quality.py`

**Interfaces:**
- Produces: `normalize_question(question: str) -> str`.
- Produces: `resolve_anchor(source_root: Path, anchor: str) -> tuple[Path, int, int, str]`.
- Produces: `validate_case_quality(cases: Sequence[Mapping[str, Any]], source_root: Path, *, max_cases_per_file: int = 15, min_source_files: int = 20) -> None`.
- Produces: `assert_preserved_cases(final_cases: Sequence[Mapping[str, Any]], historical_cases: Sequence[Mapping[str, Any]]) -> None`.
- Produces: `validate_distribution(cases: Sequence[Mapping[str, Any]]) -> None`.

- [ ] **Step 1: Implement question normalization**

```python
_NUMBERED_COMPONENT = re.compile(r"\bcomponent\s*#?\d+\b", re.IGNORECASE)

def normalize_question(question: str) -> str:
    normalized = _NUMBERED_COMPONENT.sub("component #N", question)
    normalized = re.sub(r"\b\d+\b", "N", normalized.lower())
    return " ".join(normalized.split())
```

- [ ] **Step 2: Implement anchor resolution with exact bounds**

```python
def resolve_anchor(source_root: Path, anchor: str) -> tuple[Path, int, int, str]:
    match = re.fullmatch(r"(.+):(\d+)-(\d+)", anchor)
    if match is None:
        raise ValueError(f"Invalid source anchor: {anchor}")
    relative, start_text, end_text = match.groups()
    path = source_root / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = int(start_text), int(end_text)
    if not path.is_file() or start < 1 or end < start or end > len(lines):
        raise ValueError(f"Source anchor is outside file bounds: {anchor}")
    excerpt = "\n".join(lines[start - 1 : end]).strip()
    if not excerpt:
        raise ValueError(f"Source anchor is empty: {anchor}")
    return path, start, end, excerpt
```

Check `path.is_file()` before reading in the final implementation so a missing
file produces the same controlled `ValueError`.

- [ ] **Step 3: Implement dataset-level quality checks**

Validate unique IDs, exact normalized-question uniqueness, banned numbered
components, maximum file concentration, minimum file diversity, declared-file
anchor matching, and every anchor through `resolve_anchor`.

- [ ] **Step 4: Implement preservation and distribution checks**

`assert_preserved_cases` compares the historical sequence with the prefix of the
final sequence using normal dictionary equality. `validate_distribution` requires
exactly 100 cases and `Counter(difficulty) == {"simple": 35, "medium": 35, "hard": 30}`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py -k 'rejects or historical'`

Expected: PASS.

- [ ] **Step 6: Commit the validator**

```bash
git add src/hybrid_rag/eval/gold_validation.py tests/unit/test_gold_dataset_quality.py
git commit -m "feat(eval): validate gold dataset quality"
```

### Task 3: Refactor the Generator into a Deterministic Assembler

**Files:**
- Modify: `scripts/create_100_gold_datasets.py`
- Create: `eval/gold/catalogs/llama_index_core_additions.json`
- Create: `eval/gold/catalogs/transformers_additions.json`
- Create: `eval/gold/catalogs/langchain_core_additions.json`
- Modify: `tests/unit/test_gold_dataset_quality.py`

**Interfaces:**
- Consumes: historical datasets, additions catalogs, and explicit snapshot roots.
- Produces: `assemble_dataset(historical_path: Path, additions_path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]`.
- CLI: `create_100_gold_datasets.py --llama-root PATH --transformers-root PATH --langchain-root PATH [--review-status draft|approved]`.

- [ ] **Step 1: Add a failing deterministic-assembly test**

The test assembles a temporary historical dataset plus additions twice and
asserts byte-identical canonical JSON output, preserved prefix equality, and no
mutation of input dictionaries.

- [ ] **Step 2: Replace generated loops with catalog loading**

```python
def assemble_dataset(historical_path: Path, additions_path: Path, metadata: Mapping[str, Any]):
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    additions = json.loads(additions_path.read_text(encoding="utf-8"))
    cases = [copy.deepcopy(case) for case in historical["cases"]]
    cases.extend(copy.deepcopy(case) for case in additions["cases"])
    return {**metadata, "schema_version": 1, "cases": cases}
```

- [ ] **Step 3: Validate before writing**

For each dataset, call `assert_preserved_cases`, `validate_distribution`,
`validate_case_quality`, write with `json.dumps(..., indent=2, ensure_ascii=False) + "\n"`,
then reload with `load_gold_dataset`.

- [ ] **Step 4: Keep additions catalogs empty but schema-valid for the next tasks**

Each catalog initially contains `{"cases": []}`. Generator tests use temporary
fixtures and do not claim the production outputs are complete yet.

- [ ] **Step 5: Run deterministic generator tests**

Run: `PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py -k 'assemble or deterministic'`

Expected: PASS.

- [ ] **Step 6: Commit assembler refactor**

```bash
git add scripts/create_100_gold_datasets.py eval/gold/catalogs tests/unit/test_gold_dataset_quality.py
git commit -m "refactor(eval): assemble curated gold cases"
```

### Task 4: Curate and Verify 70 New LlamaIndex Cases

**Files:**
- Modify: `eval/gold/catalogs/llama_index_core_additions.json`
- Regenerate: `eval/gold/llama_index_core_100_answer_quality.json`

**Interfaces:**
- Consumes: extracted `llama-index-core==0.14.21` snapshot matching artifact SHA-256 `4a807d31e54d066068e076eb4d066efbf95e2d2a00dcbe0eba3d9340a04cad42`.
- Produces: 70 additions with `25 simple / 25 medium / 20 hard`.

- [ ] **Step 1: Verify the wheel and extracted source snapshot**

Download only if no matching cached wheel exists. Verify the wheel SHA-256,
extract it under `/private/tmp/hybrid-rag-gold-sources/llama-index-core-0.14.21`,
and verify the deterministic source identity before reviewing cases.

- [ ] **Step 2: Select source targets across at least 20 files**

Use AST symbol inventory and `rg` to select concrete public classes and methods
from indexing, retrieval, query engines, parsing, synthesis, callbacks, agents,
storage, and evaluation. Exclude generated files and deprecated aliases whose
behavior is not defined locally.

- [ ] **Step 3: Write 25 simple additions**

Each question targets one local definition or direct behavior. Use exact source
anchors and concise source-backed contexts.

- [ ] **Step 4: Write 25 medium additions**

Each question covers a lifecycle or interaction among multiple methods in one
component. Add multiple anchors when one range cannot support the whole answer.

- [ ] **Step 5: Write 20 hard additions**

Each question covers recursive, asynchronous, cross-component, or multi-stage
behavior. Every claimed stage must be represented by an anchor and context.

- [ ] **Step 6: Generate and validate LlamaIndex output**

Run:
`PYTHONPATH=src .venv/bin/python scripts/create_100_gold_datasets.py --llama-root /private/tmp/hybrid-rag-gold-sources/llama-index-core-0.14.21 --only llama-index --review-status approved`

Expected: 100 cases, `35/35/30`, 30 preserved, zero invalid anchors.
Preservation uses the audited supplementary-v2 semantics: only AQ08
`33-170 -> 33-151` and AQ23 `232-320 -> 232-317` differ from v1, and only in
`source_anchors`.

- [ ] **Step 7: Commit the LlamaIndex catalog and output**

```bash
git add eval/gold/catalogs/llama_index_core_additions.json eval/gold/llama_index_core_100_answer_quality.json
git commit -m "feat(eval): curate 100 LlamaIndex cases"
```

### Task 5: Curate and Verify 90 New Transformers Cases

**Files:**
- Modify: `eval/gold/catalogs/transformers_additions.json`
- Regenerate: `eval/gold/transformers_100_answer_quality.json`

**Interfaces:**
- Consumes: Transformers checkout at commit `0a2757da521a7a49b8143d9e0c938f08747d682e`.
- Produces: 90 additions with `31 simple / 32 medium / 27 hard`.

- [ ] **Step 1: Prepare and verify the immutable checkout**

Use `/private/tmp/hybrid-rag-gold-sources/transformers`, verify `git rev-parse HEAD`
equals the pinned commit, and require a clean checkout.

- [ ] **Step 2: Select source targets across at least 20 files**

Cover configuration, tokenization, auto classes, model loading/saving, generation,
trainer state, pipelines, cache utilities, quantization interfaces, integrations,
and representative model components. Limit `modeling_utils.py` to at most 15 cases.

- [ ] **Step 3: Write 31 simple additions**

Target named functions/classes and direct contracts with one or more exact anchors.

- [ ] **Step 4: Write 32 medium additions**

Target state transitions and interactions among helpers inside one subsystem.

- [ ] **Step 5: Write 27 hard additions**

Target checkpoint dispatch, generation loops, trainer lifecycle, pipeline batching,
or cross-file auto-class resolution with complete evidence paths.

- [ ] **Step 6: Generate and validate Transformers output**

Run the generator with `--only transformers --review-status approved` and the
pinned source root. Expected: 100 cases, `35/35/30`, ten preserved, zero invalid
anchors, at least 20 source files, no file above 15 cases.

- [ ] **Step 7: Commit the Transformers catalog and output**

```bash
git add eval/gold/catalogs/transformers_additions.json eval/gold/transformers_100_answer_quality.json
git commit -m "feat(eval): curate 100 Transformers cases"
```

### Task 6: Curate and Verify 90 New LangChain Core Cases

**Files:**
- Modify: `eval/gold/catalogs/langchain_core_additions.json`
- Regenerate: `eval/gold/langchain_core_100_answer_quality.json`

**Interfaces:**
- Consumes: LangChain checkout at commit `51578289bb1f696a643e0740be1441039d8af8ce`, source root `libs/core`.
- Produces: 90 additions with `31 simple / 32 medium / 27 hard`.

- [ ] **Step 1: Prepare and verify the immutable checkout**

Use `/private/tmp/hybrid-rag-gold-sources/langchain`, verify the pinned commit,
require a clean checkout, and pass `libs/core` as the source root.

- [ ] **Step 2: Select source targets across at least 20 files**

Cover Runnable composition, configuration, callbacks, messages, prompts,
retrievers, output parsers, tools, documents, vector-store interfaces, language
model abstractions, serialization, rate limiting, caches, and tracing. Limit
`runnables/base.py` to at most 15 cases.

- [ ] **Step 3: Write 31 simple additions**

Target concrete classes and direct synchronous or asynchronous behavior.

- [ ] **Step 4: Write 32 medium additions**

Target lifecycle, configuration propagation, or streaming behavior within one
subsystem.

- [ ] **Step 5: Write 27 hard additions**

Target composition, callback propagation, fallback/retry, streaming, or
cross-component serialization behavior with multiple anchors when needed.

- [ ] **Step 6: Generate and validate LangChain output**

Run the generator with `--only langchain-core --review-status approved` and the
pinned `libs/core` root. Expected: 100 cases, `35/35/30`, ten preserved, zero
invalid anchors, at least 20 source files, no file above 15 cases.

- [ ] **Step 7: Commit the LangChain catalog and output**

```bash
git add eval/gold/catalogs/langchain_core_additions.json eval/gold/langchain_core_100_answer_quality.json
git commit -m "feat(eval): curate 100 LangChain cases"
```

### Task 7: Final Audit, Documentation, and Regression Verification

**Files:**
- Modify: `docs/evaluation.md`
- Modify: `tests/unit/test_answer_quality.py`
- Modify: `tests/unit/test_gold_dataset_quality.py`

**Interfaces:**
- Consumes: all three final datasets and pinned source roots.
- Produces: documented supplementary evidence scope and complete regression coverage.

- [ ] **Step 1: Extend checked-in dataset tests to all three 100-case files**

Parameterize the production-loader test over all final datasets and assert case
count, review grade, source identity, and `35/35/30` distribution.

- [ ] **Step 2: Run the final source audit**

Run the quality test with snapshot roots supplied through `GOLD_LLAMA_ROOT`,
`GOLD_TRANSFORMERS_ROOT`, and `GOLD_LANGCHAIN_ROOT`. Expected report: 300 total cases, 50 preserved historical
cases, zero normalized duplicates, zero invalid anchors, and all diversity limits met.

- [ ] **Step 3: Verify generator determinism**

Run the generator twice into separate temporary directories and compare
`shasum -a 256` for all three outputs. Expected: identical hashes.

- [ ] **Step 4: Update evaluation documentation**

Document that the original 30-case LlamaIndex result remains the submitted thesis
evidence, while the three 100-case datasets are post-submission supplementary
replications. State `approved_ai_source_review` explicitly and do not imply human review.

- [ ] **Step 5: Run full verification**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/unit/test_gold_dataset_quality.py tests/unit/test_answer_quality.py
PYTHONPATH=src .venv/bin/pytest -q tests/unit
git diff --check
git status --short
```

Expected: all tests pass; only intentional implementation files are changed before
the final commit; historical dataset hashes match Task 1.

- [ ] **Step 6: Commit documentation and final tests**

```bash
git add docs/evaluation.md tests/unit/test_answer_quality.py tests/unit/test_gold_dataset_quality.py
git commit -m "docs(eval): document 100-case replications"
```

- [ ] **Step 7: Review branch history and final diff**

Run `git log --oneline --decorate -10` and `git diff develop...HEAD --stat`.
Confirm no presentation or submitted-thesis files changed.
