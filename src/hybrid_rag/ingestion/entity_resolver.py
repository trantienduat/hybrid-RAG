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
from typing import Any

from hybrid_rag.ingestion.parser import ParseResult


def resolve(result: ParseResult) -> ParseResult:
    """
    Merge external stub nodes (Module, Class, Function) into real nodes where possible.

    Steps:
    1. Build map: module_stem → real Module node id
    2. Build map: class_simple_name → real Class node id  (cross-file class linking)
    3. Build map: function_simple_name → list of real Function node ids
    4. Resolve stub nodes (Module/Class stubs → real Module/Class nodes)
    5. Resolve __call__ call-edge stubs via Tiers A–D:
       - Tier A: Sibling method in class context
       - Tier B: Same module function resolution
       - Tier C: Import-based function resolution
       - Tier D: Global unique function fallback (heuristic for single-occurrence names)
    6. Rewrite all edges src/dst referencing a stub → real id
    7. Drop stub nodes that were fully resolved
    """
    # Separate real vs stub nodes
    real_modules: dict[str, str] = {}  # stem → real node id
    real_classes: dict[str, str] = {}  # simple name → real node id
    real_functions: dict[str, list[str]] = {}  # simple name → list of FQNs
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
        elif node.label == "Class":
            if node.properties.get("type") == "external":
                stub_ids.add(node.id)
            else:
                class_nid = node.id.split("::")[-1]
                if "." in class_nid:
                    class_nid = class_nid.split(".")[-1]
                simple_name = node.properties.get("name", class_nid)
                # First definition wins (avoids ambiguity in large repos)
                real_classes.setdefault(simple_name, node.id)
        elif node.label == "Function":
            if node.properties.get("type") == "external":
                stub_ids.add(node.id)
            else:
                simple_name = node.properties.get("name", node.id.split(".")[-1])
                real_functions.setdefault(simple_name, []).append(node.id)

    # Build redirect map: stub_id → real_id
    redirect: dict[str, str] = {}

    # Module / Class stubs → real nodes
    for stub_id in stub_ids:
        stem = stub_id.split(".")[-1]
        if stem in real_modules:
            redirect[stub_id] = real_modules[stem]
        elif stub_id in real_modules:
            redirect[stub_id] = real_modules[stub_id]
        elif stub_id in real_classes:
            redirect[stub_id] = real_classes[stub_id]
        elif stem in real_classes:
            redirect[stub_id] = real_classes[stem]

    # Resolve __call__ stubs in CALLS edges
    _resolve_calls(result, real_modules, real_functions, redirect)

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
        n
        for n in resolved.nodes
        if not (
            n.id in redirect
            and (
                n.properties.get("type") == "external"
                or n.id.startswith("__call__")
            )
        )
    ]

    # Rewrite edges
    for edge in resolved.edges:
        if edge.src_id in redirect:
            edge.src_id = redirect[edge.src_id]
        if edge.dst_id in redirect:
            edge.dst_id = redirect[edge.dst_id]

    return resolved


def _resolve_calls(
    result: ParseResult,
    real_modules: dict[str, str],
    real_functions: dict[str, list[str]],
    redirect: dict[str, str],
) -> None:
    """Helper to resolve __call__ stubs in CALLS edges across Tiers A–D."""
    module_imports: dict[str, set[str]] = {}
    for edge in result.edges:
        if edge.rel == "IMPORTS":
            module_imports.setdefault(edge.src_id, set()).add(edge.dst_id.split(".")[-1])

    for edge in result.edges:
        if edge.rel == "CALLS" and edge.dst_id.startswith("__call__"):
            stub_id = edge.dst_id
            callee_name = stub_id[len("__call__") :]
            caller_id = edge.src_id
            parts = caller_id.split(".")
            resolved_id = None

            # Tier A. Sibling method in class context
            if len(parts) >= 3:
                class_fqn = ".".join(parts[:-1])
                target_method_fqn = f"{class_fqn}.{callee_name}"
                if any(n.id == target_method_fqn and n.label == "Function" for n in result.nodes):
                    resolved_id = target_method_fqn

            # Tier B. Same module resolution
            if not resolved_id:
                module_fqn = None
                for m_id in real_modules.values():
                    if caller_id.startswith(m_id):
                        if module_fqn is None or len(m_id) > len(module_fqn):
                            module_fqn = m_id
                if module_fqn:
                    target_func_fqn = f"{module_fqn}.{callee_name}"
                    if any(n.id == target_func_fqn and n.label == "Function" for n in result.nodes):
                        resolved_id = target_func_fqn

            # Tier C. Import-based resolution
            if not resolved_id and module_fqn:
                callee_expr = edge.properties.get("callee_expr", "")
                imports = module_imports.get(module_fqn, set())
                if callee_name in imports:
                    candidates = real_functions.get(callee_name, [])
                    if len(candidates) == 1:
                        resolved_id = candidates[0]
                    elif len(candidates) > 1:
                        matching_candidates = []
                        for cand in candidates:
                            cand_parts = cand.split(".")
                            for imp in imports:
                                if imp in cand_parts:
                                    matching_candidates.append(cand)
                                    break
                        if len(matching_candidates) == 1:
                            resolved_id = matching_candidates[0]
                elif "." in callee_expr:
                    prefix = callee_expr.split(".")[0]
                    if prefix in imports:
                        candidates = real_functions.get(callee_name, [])
                        for cand in candidates:
                            if cand.endswith(f".{prefix}.{callee_name}") or cand.endswith(
                                f"{prefix}.{callee_name}"
                            ):
                                resolved_id = cand
                                break

            # Tier D. Global fallback (only if unique)
            if not resolved_id:
                candidates = real_functions.get(callee_name, [])
                if len(candidates) == 1:
                    resolved_id = candidates[0]

            if resolved_id:
                redirect[stub_id] = resolved_id


def stub_count(result: ParseResult) -> int:
    """Return number of unresolved external stub nodes (Module, Class, Function)."""
    return sum(
        1
        for n in result.nodes
        if n.properties.get("type") == "external" or n.id.startswith("__call__")
    )


def resolve_global(result: ParseResult, graph_store: Any) -> ParseResult:
    """
    Query FalkorDB to resolve remaining external stubs against real nodes in previously indexed repositories.

    Enables Multi-Repo cross-linking: (RepoB.Class)-[:INHERITS/CALLS]->(RepoA.Class).
    Uses a batched Cypher query to avoid N+1 query overhead.
    """
    external_stubs = [
        node.id
        for node in result.nodes
        if node.properties.get("type") == "external" or node.id.startswith("__call__")
    ]
    if not external_stubs:
        return result

    redirect: dict[str, str] = {}

    # Query FalkorDB in a single batched query
    query = (
        "UNWIND $stubs AS stub "
        "MATCH (n) WHERE (n.id = stub OR n.name = stub OR n.id ENDS WITH '.' + stub) "
        "AND (n.type IS NULL OR n.type <> 'external') "
        "RETURN stub, n.id"
    )
    try:
        res = graph_store.query(query, {"stubs": external_stubs})
        if res.result_set:
            for row in res.result_set:
                stub_id, real_id = row[0], row[1]
                redirect[stub_id] = real_id
    except Exception:
        # Skip gracefully if database isn't initialized or query fails
        pass

    if not redirect:
        return result

    resolved = ParseResult(
        nodes=copy.deepcopy(result.nodes),
        edges=copy.deepcopy(result.edges),
        errors=list(result.errors),
    )

    resolved.nodes = [
        n
        for n in resolved.nodes
        if not (
            n.id in redirect
            and (n.properties.get("type") == "external" or n.id.startswith("__call__"))
        )
    ]

    for edge in resolved.edges:
        if edge.src_id in redirect:
            edge.src_id = redirect[edge.src_id]
        if edge.dst_id in redirect:
            edge.dst_id = redirect[edge.dst_id]

    return resolved
