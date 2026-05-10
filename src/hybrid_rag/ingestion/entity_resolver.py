"""
Entity resolver: deduplicates stub Module nodes against real KG nodes.

When the parser sees `import falkordb` in repo code, it creates a stub
  Module(id="falkordb", type="external")

If the codebase actually contains a file that resolves to that same module
name (e.g. `src/falkordb/__init__.py`), all IMPORTS edges pointing at the
stub should instead point at the real Module node.

This is a lightweight heuristic resolver — it does NOT do full package
resolution. It matches stubs by stem name against real module names.

Public API:
  resolve(result: ParseResult) -> ParseResult
      Returns a NEW ParseResult with stubs merged and edges rewritten.
"""
from __future__ import annotations

import copy
from pathlib import Path

from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult


def resolve(result: ParseResult) -> ParseResult:
    """
    Merge external stub Module nodes into real Module nodes where possible.

    Steps:
    1. Build map: module_stem → real Module node id
    2. For each stub Module whose stem matches a real module, record a redirect
    3. Rewrite all edges src/dst that reference the stub → real id
    4. Drop stub nodes that were fully resolved
    """
    # Separate real vs stub nodes
    real_modules: dict[str, str] = {}   # stem → real node id
    stub_ids: set[str] = set()

    for node in result.nodes:
        if node.label == "Module":
            if node.properties.get("type") == "external":
                stub_ids.add(node.id)
            else:
                # Real module: index by stem (filename without extension)
                stem = Path(node.properties.get("file_path", node.id)).stem
                real_modules[stem] = node.id
                # Also index by the last component of dotted module path
                name = node.properties.get("name", stem)
                real_modules.setdefault(name, node.id)

    # Build redirect map: stub_id → real_id (only where a match exists)
    redirect: dict[str, str] = {}
    for stub_id in stub_ids:
        # stub id is typically the bare package name, e.g. "falkordb"
        stem = stub_id.split(".")[-1]
        if stem in real_modules:
            redirect[stub_id] = real_modules[stem]
        elif stub_id in real_modules:
            redirect[stub_id] = real_modules[stub_id]

    if not redirect:
        return result  # nothing to resolve, return unchanged

    # Deep-copy so original is untouched
    resolved = ParseResult(
        nodes=copy.deepcopy(result.nodes),
        edges=copy.deepcopy(result.edges),
        errors=list(result.errors),
    )

    # Drop resolved stubs from node list
    resolved.nodes = [
        n for n in resolved.nodes
        if not (n.label == "Module" and n.id in redirect)
    ]

    # Rewrite edges
    for edge in resolved.edges:
        if edge.src_id in redirect:
            edge.src_id = redirect[edge.src_id]
        if edge.dst_id in redirect:
            edge.dst_id = redirect[edge.dst_id]

    return resolved


def stub_count(result: ParseResult) -> int:
    """Return number of unresolved external stub Module nodes."""
    return sum(
        1 for n in result.nodes
        if n.label == "Module" and n.properties.get("type") == "external"
    )
