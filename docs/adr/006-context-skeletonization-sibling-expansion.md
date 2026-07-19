# ADR-006: Context Skeletonization with Sibling Body Expansion

## Status
Accepted

## Context
When assembling codebase context for prompt packaging, the system previously retrieved full, uncollapsed code bodies of the target functions/methods, as well as their neighboring nodes. This resulted in:
1. **High Token Bloat**: Sending full sibling code bodies wasted 60–80% of prompt context space.
2. **Context Dilution**: The LLM was overwhelmed by logic from unrelated functions inside retrieved classes.

To resolve this, we introduced **Basic Skeletonization** (Phase 1) using Tree-sitter to collapse sibling/neighbor methods to `...`, retaining only signatures and docstrings. However, this introduced a new problem: **Sibling Body Loss**, where important sibling methods called directly by the target method were also collapsed, depriving the LLM of necessary logic.

## Decision
Implement **Graph-based Sibling Body Expansion** on top of the context skeletonizer.
When a focus method is retrieved:
1. Query FalkorDB for any sibling methods in the same file that are directly called (`CALLS` relationship) by the focus method.
2. Proactively add these called sibling methods to the uncollapsed `focus_names` list.

## Rationale
* **Maximizes Accuracy**: Preserves the complete code bodies of siblings that are actually part of the target method's execution path.
* **Saves Tokens**: Continues to collapse unrelated sibling methods, keeping token usage minimal.
* **Low Latency**: Querying direct called sibling names takes $<5\text{ms}$ in FalkorDB.
* **Backward Compatible**: Falls back gracefully to basic skeletonization if no calls exist or the graph database is inaccessible.

## Consequences
* Context assembly now performs a small Cypher query per retrieved function node.
* Prompt text remains syntactically coherent and compiles cleanly.
