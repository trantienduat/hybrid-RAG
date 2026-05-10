# ADR-002: AST Parser — PENDING (Spike Required)

## Status
Pending — see `experiments/spike-01-ast-parser.ipynb`

## Context
Triplet extraction requires parsing Python (and optionally Java) source into structured AST nodes.

Candidates:
- **Python `ast` module** — stdlib, zero install, Python-only, stable API
- **tree-sitter** — multi-language (Python + Java in one library), incremental parsing, widely used in editors
- **LibCST** — concrete syntax tree, preserves formatting, Python-only

## Decision
TBD after spike.

## Spike Questions to Answer
1. Does tree-sitter parse all LlamaIndex files without error?
2. Can we extract all 4 node types (Module/Class/Function/Variable) reliably from both?
3. Performance: time to parse 100 files each approach?
4. Is Java support actually needed for M1, or Python-first is sufficient?

## Preliminary Leaning
tree-sitter — due to multi-language and LlamaIndex `CodeSplitter` already uses it internally.

## Update This ADR After Spike
Replace "PENDING" status with "Accepted" and fill Decision + Consequences sections.
