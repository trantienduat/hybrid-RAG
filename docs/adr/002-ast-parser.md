# ADR-002: AST Parser — tree-sitter

## Status
Accepted — spike completed 2026-05-10, see `experiments/spike_01_ast_parser.py`

## Context
Triplet extraction requires parsing Python (and optionally Java) source into structured AST nodes.

Candidates evaluated:
- **Python `ast` module** — stdlib, zero install, Python-only, stable API
- **tree-sitter** — multi-language (Python + Java), incremental parsing, widely used in editors
- **LibCST** — concrete syntax tree, Python-only (eliminated early)

## Spike Results (100 stdlib .py files)

| Metric | ast | tree-sitter |
|--------|-----|-------------|
| Parse errors | 0 | 0 |
| Avg ms/file | 1.37 | 1.18 |
| Java support | No | Yes |
| Multi-language | No | Yes |

## Decision
Use **tree-sitter** (`tree-sitter-python` + `tree-sitter-java`).

## Rationale
- Equal reliability (0 errors both), tree-sitter 14% faster
- Java support future-proofs M2+ without architecture change
- LlamaIndex `CodeSplitter` uses tree-sitter internally — consistent toolchain
- Node-type queries (`class_definition`, `function_definition`) map directly to KG schema

## Consequences
- Deps: `tree-sitter`, `tree-sitter-python`, `tree-sitter-java`
- Must use tree-sitter field names (differ from Python `ast` attribute names)
- Incremental parsing available if file-watching added later
