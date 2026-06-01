"""
Triplet extractor: converts ParseResult → (subject, predicate, object) triples
ready for FalkorDB ingestion.

Each triple is a dict with:
  - src_id, src_label: FalkorDB node key + label
  - rel: relationship type (IMPORTS, DEFINES, INHERITS, CALLS, USES, DEFINED_IN)
  - dst_id, dst_label: target node key + label
  - properties: dict of edge properties
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hybrid_rag.ingestion.parser import ParseResult

# Label inference map — used to resolve stub nodes to a label
_STUB_LABEL: dict[str, str] = {
    "external": "Module",
}

# Default labels for stub node types we know about
_REL_DST_LABEL: dict[str, str] = {
    "IMPORTS": "Module",
    "DEFINES": None,   # resolved per-edge from node registry
    "INHERITS": "Class",
    "CALLS": "Function",
    "USES": "Variable",
    "DEFINED_IN": "Module",
}


@dataclass
class Triple:
    src_id: str
    src_label: str
    rel: str
    dst_id: str
    dst_label: str
    properties: dict[str, Any] = field(default_factory=dict)


def extract_triples(result: ParseResult) -> list[Triple]:
    """Convert ParseResult into a flat list of Triples."""
    # Build id→label lookup from all nodes in result
    id_to_label: dict[str, str] = {n.id: n.label for n in result.nodes}

    triples: list[Triple] = []
    for edge in result.edges:
        src_label = id_to_label.get(edge.src_id, _infer_label(edge.src_id, edge.rel, src=True))
        dst_label = id_to_label.get(edge.dst_id, _infer_label(edge.dst_id, edge.rel, src=False))

        triples.append(Triple(
            src_id=edge.src_id,
            src_label=src_label,
            rel=edge.rel,
            dst_id=edge.dst_id,
            dst_label=dst_label,
            properties=edge.properties,
        ))
    return triples


def _infer_label(node_id: str, rel: str, *, src: bool) -> str:
    """Best-effort label for a node not in the current parse registry."""
    if rel == "CALLS" and not src:
        return "Function"
    if rel == "IMPORTS" and not src:
        return "Module"
    if rel == "INHERITS" and not src:
        return "Class"
    if rel == "DEFINED_IN" and not src:
        return "Module"
    if rel == "DEFINES" and src:
        return "Module"  # could be Class, but Module is the safe fallback
    return "Module"
