# Source-Reviewed 100-Case Gold Datasets

## Goal

Replace the low-quality generated 100-case answer-quality datasets with three
source-backed datasets suitable for supplementary cross-repository evaluation,
without changing the 30 LlamaIndex cases already used in the submitted thesis.

## Scope

The deliverable covers these repositories and immutable source snapshots:

- `llama-index-core==0.14.21`, source identity
  `sha256:13b55bcf111885b7c2c9c07886b5656769041331c81c28bcf3cbfb87939ec509`.
- Transformers commit `0a2757da521a7a49b8143d9e0c938f08747d682e`.
- LangChain Core at LangChain commit
  `51578289bb1f696a643e0740be1441039d8af8ce`, subdirectory `libs/core`.

The existing thesis dataset
`eval/gold/llama_index_core_answer_quality_v1.json` must remain byte-for-byte
unchanged. The existing ten-case Transformers and LangChain datasets must also
remain unchanged as historical source-reviewed baselines.

## Dataset Composition

Each new dataset contains exactly 100 unique cases with 35 simple, 35 medium,
and 30 hard cases.

- LlamaIndex: copy the 30 thesis cases exactly, preserving their IDs and case
  contents except for the two audited supplementary-v2 source-anchor corrections
  below, then add 70 source-reviewed cases with IDs `AQ31` through `AQ100`.
- Transformers: copy the ten v1 cases exactly, preserving IDs and case contents,
  then add 90 source-reviewed cases with IDs `transformers-t11` through
  `transformers-t100`.
- LangChain Core: copy the ten v1 cases exactly, preserving IDs and case
  contents, then add 90 source-reviewed cases with IDs `langchain-l11` through
  `langchain-l100`.

The final difficulty counts apply to the complete 100-case dataset. Preserved
cases retain their existing difficulty labels; new cases fill the remaining
counts.

### Supplementary-v2 provenance corrections

Source validation against the pinned `llama-index-core==0.14.21` wheel found two
historical anchors whose end lines extend past the real files. The submitted v1
fixture remains byte-for-byte unchanged. Only the assembled supplementary v2
output corrects these provenance ranges:

- `AQ08`: `response_synthesizers/factory.py:33-170` becomes
  `response_synthesizers/factory.py:33-151`.
- The second `AQ23` anchor, `retrievers/fusion_retriever.py:232-320`, becomes
  `retrievers/fusion_retriever.py:232-317`.

All other fields in the first 30 cases, including IDs, difficulties, questions,
answers, source files, contexts, and all other anchors, remain exactly equal to
the historical fixture. The assembler fails if either expected old anchor is
missing or duplicated; generic source validation is unchanged.

## Case Quality Contract

Every case must satisfy all of the following:

1. The question names a concrete class, function, method, enum, data structure,
   or lifecycle. Generic labels such as `component #N` are forbidden.
2. The reference answer is supported by the pinned source snapshot and does not
   claim behavior outside the declared source anchors.
3. Every source anchor resolves to an existing declared source file and a valid,
   non-empty line range in that snapshot.
4. Reference contexts are concise source-backed evidence summaries. A case may
   use multiple contexts and anchors when its answer combines multiple paths.
5. Cases must be semantically distinct after normalizing IDs, numbers, and
   whitespace. Changing only a number does not create a new case.
6. Difficulty represents evidence requirements:
   - simple: one local definition or direct behavior;
   - medium: one lifecycle or multiple methods in one component;
   - hard: cross-component, recursive, asynchronous, or multi-stage behavior.
7. No source file may provide more than 15 cases in one dataset. At least 20
   distinct source files must be represented per repository unless the pinned
   source snapshot contains fewer eligible files.

## Source Acquisition and Verification

Source verification operates only on immutable local snapshots. The generator
accepts explicit snapshot roots rather than downloading a moving branch. It
checks that each snapshot identity matches the dataset identity before writing
approved output.

For each case, validation reads the declared source file, verifies anchor bounds,
and records the anchored text for audit. Dataset generation fails if a file or
line range is missing, if an anchor is empty, or if a preserved case differs
from its historical dataset.

## Generator Design

`scripts/create_100_gold_datasets.py` becomes a deterministic assembler and
validator rather than a source of numbered filler questions. Curated case
catalogs remain explicit, reviewable data in the script or focused catalog files.
The generator:

1. loads and fingerprints the historical datasets;
2. copies preserved cases unchanged;
3. appends curated new cases;
4. validates counts, IDs, difficulty distribution, semantic uniqueness, source
   diversity, anchor bounds, and source identities;
5. writes JSON deterministically;
6. reloads each file through the production gold-dataset loader.

## Review State

Generated datasets remain `draft` until every new case passes source-anchor
validation and the review checklist. Once that evidence exists, the generator
may emit `approved` with `reviewer_type: ai_source_review`. This grade must not
be presented as human validation or as confirmatory thesis evidence.

## Tests and Acceptance Criteria

Automated tests must prove:

- all historical datasets are byte-for-byte unchanged;
- each new dataset has 100 unique cases and the exact `35/35/30` distribution;
- all preserved cases are identical to their historical versions;
- no normalized template group contains more than one case;
- source-file concentration and diversity limits pass;
- every source anchor exists and is within file bounds in its pinned snapshot;
- the production loader accepts all three datasets;
- generator output is deterministic across two runs.

The work is complete only when the tests pass and a final audit reports zero
placeholder questions, zero invalid anchors, and zero changes to the thesis
30-case file.
