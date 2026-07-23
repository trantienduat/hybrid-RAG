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
    Merge external stub nodes into real nodes where possible.

    Steps:
    1. Build map: module_stem → real Module node id
    2. Build map: class_simple_name → real Class node id  (M2: cross-file class linking)
    3. For stubs that match a real node, record a redirect
    4. Rewrite all edges src/dst that reference a stub → real id
    5. Drop stub nodes that were fully resolved
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
            class_nid = node.id.split("::")[-1]
            if "." in class_nid:
                class_nid = class_nid.split(".")[-1]
            simple_name = node.properties.get("name", class_nid)
            # First definition wins (avoids ambiguity in large repos)
            real_classes.setdefault(simple_name, node.id)
        elif node.label == "Function":
            simple_name = node.properties.get("name", node.id.split(".")[-1])
            real_functions.setdefault(simple_name, []).append(node.id)

    # Build redirect map: stub_id → real_id
    redirect: dict[str, str] = {}

    # Module stubs → real Module nodes (existing M1 behaviour)
    for stub_id in stub_ids:
        stem = stub_id.split(".")[-1]
        if stem in real_modules:
            redirect[stub_id] = real_modules[stem]
        elif stub_id in real_modules:
            redirect[stub_id] = real_modules[stub_id]
        # M2: class name stubs used in INHERITS/USES edges (stored as Module stubs
        # because _ensure_stub always creates Module stubs)
        elif stub_id in real_classes:
            redirect[stub_id] = real_classes[stub_id]
        elif stem in real_classes:
            redirect[stub_id] = real_classes[stem]

    # Resolve __call__ stubs in CALLS edges
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

            # A. Sibling method in class context
            if len(parts) >= 3:
                class_fqn = ".".join(parts[:-1])
                target_method_fqn = f"{class_fqn}.{callee_name}"
                if any(n.id == target_method_fqn and n.label == "Function" for n in result.nodes):
                    resolved_id = target_method_fqn

            # B. Same module resolution
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

            # C. Import-based resolution
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

            # D. Global fallback (only if unique)
            if not resolved_id:
                candidates = real_functions.get(callee_name, [])
                if len(candidates) == 1:
                    resolved_id = candidates[0]

            if resolved_id:
                redirect[stub_id] = resolved_id

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
                (n.label == "Module" and n.properties.get("type") == "external")
                or (n.label == "Function" and n.properties.get("type") == "external")
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


def stub_count(result: ParseResult) -> int:
    """Return number of unresolved external stub Module nodes."""
    return sum(
        1 for n in result.nodes if n.label == "Module" and n.properties.get("type") == "external"
    )


def resolve_global(result: ParseResult, graph_store: Any) -> ParseResult:
    """
    Query FalkorDB to resolve remaining external stubs against real nodes in previously indexed repositories.

    Enables Multi-Repo cross-linking: (RepoB.Class)-[:INHERITS/CALLS]->(RepoA.Class).
    """
    # 1. Gather unresolved external module stubs in the current parse results
    external_stubs = [
        node.id
        for node in result.nodes
        if node.label == "Module" and node.properties.get("type") == "external"
    ]
    if not external_stubs:
        return result

    redirect: dict[str, str] = {}

    # 2. Query FalkorDB to check if a real node matching that FQN ID already exists
    for stub_id in external_stubs:
        # Check if there is a real (non-external) node matching the stub's FQN ID in FalkorDB
        query = (
            "MATCH (n) WHERE (n.id = $stub_id OR n.name = $stub_id OR n.id ENDS WITH $ends_with) "
            "AND (n.type IS NULL OR n.type <> 'external') RETURN n.id LIMIT 1"
        )
        ends_with = f".{stub_id}"
        try:
            res = graph_store.query(query, {"stub_id": stub_id, "ends_with": ends_with})
            if res.result_set:
                row = res.result_set[0]
                real_id = row[0]
                redirect[stub_id] = real_id
        except Exception:
            # Skip gracefully if database isn't initialized or query fails
            continue

    if not redirect:
        return result

    # 3. Create a deep copy to avoid modifying original results in-place
    resolved = ParseResult(
        nodes=copy.deepcopy(result.nodes),
        edges=copy.deepcopy(result.edges),
        errors=list(result.errors),
    )

    # 4. Remove resolved stubs from the node list
    resolved.nodes = [
        n
        for n in resolved.nodes
        if not (n.label == "Module" and n.id in redirect and n.properties.get("type") == "external")
    ]

    # 5. Rewrite all edges in-place
    for edge in resolved.edges:
        if edge.src_id in redirect:
            edge.src_id = redirect[edge.src_id]
        if edge.dst_id in redirect:
            edge.dst_id = redirect[edge.dst_id]

    return resolved
