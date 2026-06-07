"""
Merger: combines supplemental edges (from LLM extraction) into a ParseResult.

Public API:
  merge_supplemental(base: ParseResult, extra_edges: list[EdgeData]) -> ParseResult
      Returns a NEW ParseResult. The original is never mutated.
      - Deduplicates edges by (src_id, rel, dst_id)
      - Adds minimal stub nodes for any endpoint not already in base
"""

from __future__ import annotations

import copy

from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult


def merge_supplemental(base: ParseResult, extra_edges: list[EdgeData]) -> ParseResult:
    """
    Return a new ParseResult with extra_edges merged in.

    Deduplicates edges; adds stub nodes for unknown endpoints so the graph
    remains structurally consistent. The entity resolver should be run after
    merge to resolve those stubs to real nodes.
    """
    if not extra_edges:
        return base  # fast path — no copy needed

    merged = ParseResult(
        nodes=copy.deepcopy(base.nodes),
        edges=copy.deepcopy(base.edges),
        errors=list(base.errors),
    )

    existing_node_ids: set[str] = {n.id for n in merged.nodes}
    existing_edges: set[tuple[str, str, str]] = {(e.src_id, e.rel, e.dst_id) for e in merged.edges}

    for edge in extra_edges:
        key = (edge.src_id, edge.rel, edge.dst_id)
        if key in existing_edges:
            continue

        # Ensure both endpoints have node entries (add stubs for unknowns)
        for node_id in (edge.src_id, edge.dst_id):
            if node_id not in existing_node_ids:
                merged.nodes.append(_make_stub(node_id))
                existing_node_ids.add(node_id)

        merged.edges.append(copy.deepcopy(edge))
        existing_edges.add(key)

    return merged


def _make_stub(node_id: str) -> NodeData:
    """
    Create a minimal stub node for an unknown endpoint.

    Infers label heuristically:
    - "__call__*"  → Function stub
    - Otherwise    → Module stub (entity resolver will upgrade to Class if matched)
    """
    if node_id.startswith("__call__"):
        label = "Function"
        name = node_id[len("__call__") :]
    else:
        label = "Module"
        name = node_id

    return NodeData(
        label=label,
        id=node_id,
        properties={"name": name, "file_path": "", "type": "external"},
    )
