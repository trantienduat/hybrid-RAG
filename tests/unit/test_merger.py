"""
Unit tests for ingestion/merger.py.
No external services required.
"""

from __future__ import annotations

from hybrid_rag.ingestion.merger import merge_supplemental
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult

# ── helpers ───────────────────────────────────────────────────────────────────


def _node(label: str, node_id: str) -> NodeData:
    return NodeData(label=label, id=node_id, properties={"name": node_id})


def _edge(src: str, rel: str, dst: str, **props) -> EdgeData:
    return EdgeData(src_id=src, rel=rel, dst_id=dst, properties=dict(props))


def _base() -> ParseResult:
    return ParseResult(
        nodes=[
            _node("Module", "a.py"),
            _node("Function", "a.py::foo"),
        ],
        edges=[_edge("a.py", "DEFINES", "a.py::foo")],
    )


# ── basic merge behaviour ─────────────────────────────────────────────────────


class TestMergeSupplemental:
    def test_empty_extras_returns_base_unchanged(self):
        base = _base()
        merged = merge_supplemental(base, [])
        assert merged is base  # fast path

    def test_new_edge_added(self):
        base = _base()
        extra = [_edge("a.py::foo", "USES", "SomeClass")]
        merged = merge_supplemental(base, extra)
        assert any(e.rel == "USES" for e in merged.edges)

    def test_stub_node_created_for_unknown_dst(self):
        base = _base()
        extra = [_edge("a.py::foo", "USES", "SomeClass")]
        merged = merge_supplemental(base, extra)
        ids = {n.id for n in merged.nodes}
        assert "SomeClass" in ids

    def test_call_stub_gets_function_label(self):
        base = _base()
        extra = [_edge("a.py::foo", "CALLS", "__call__bar")]
        merged = merge_supplemental(base, extra)
        stub = next(n for n in merged.nodes if n.id == "__call__bar")
        assert stub.label == "Function"

    def test_regular_stub_gets_module_label(self):
        base = _base()
        extra = [_edge("a.py::foo", "USES", "SomeClass")]
        merged = merge_supplemental(base, extra)
        stub = next(n for n in merged.nodes if n.id == "SomeClass")
        assert stub.label == "Module"

    def test_duplicate_edge_not_added(self):
        base = _base()
        extra = [
            _edge("a.py", "DEFINES", "a.py::foo"),  # already exists
        ]
        merged = merge_supplemental(base, extra)
        # Should have exactly same number of edges as base
        defines_edges = [e for e in merged.edges if e.rel == "DEFINES"]
        assert len(defines_edges) == 1

    def test_two_identical_extras_deduplicated(self):
        base = _base()
        extra = [
            _edge("a.py::foo", "USES", "SomeClass"),
            _edge("a.py::foo", "USES", "SomeClass"),
        ]
        merged = merge_supplemental(base, extra)
        uses_edges = [e for e in merged.edges if e.rel == "USES"]
        assert len(uses_edges) == 1

    def test_original_not_mutated(self):
        base = _base()
        original_node_count = len(base.nodes)
        original_edge_count = len(base.edges)
        merge_supplemental(base, [_edge("a.py::foo", "USES", "X")])
        assert len(base.nodes) == original_node_count
        assert len(base.edges) == original_edge_count

    def test_existing_node_not_duplicated_by_stub(self):
        """If a.py::foo is already a node, adding an edge with src=a.py::foo should not create a stub."""
        base = _base()
        extra = [_edge("a.py::foo", "USES", "NewClass")]
        merged = merge_supplemental(base, extra)
        # a.py::foo should still appear exactly once
        foo_nodes = [n for n in merged.nodes if n.id == "a.py::foo"]
        assert len(foo_nodes) == 1

    def test_multiple_new_edges(self):
        base = _base()
        extra = [
            _edge("a.py::foo", "USES", "ClassA"),
            _edge("a.py::foo", "USES", "ClassB"),
            _edge("a.py::foo", "CALLS", "__call__helper"),
        ]
        merged = merge_supplemental(base, extra)
        new_rels = {e.rel for e in merged.edges}
        assert "USES" in new_rels
        assert "CALLS" in new_rels
        uses_edges = [e for e in merged.edges if e.rel == "USES"]
        assert len(uses_edges) == 2

    def test_edge_properties_preserved(self):
        base = _base()
        extra = [_edge("a.py::foo", "USES", "X", source="llm", confidence=0.9)]
        merged = merge_supplemental(base, extra)
        uses_edge = next(e for e in merged.edges if e.rel == "USES")
        assert uses_edge.properties["source"] == "llm"
        assert uses_edge.properties["confidence"] == 0.9
