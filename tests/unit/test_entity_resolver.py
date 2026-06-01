"""
Unit tests for ingestion/entity_resolver.py — including M2 cross-file class linking.
No external services required.
"""
from __future__ import annotations

from hybrid_rag.ingestion.entity_resolver import resolve, stub_count
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult

# ── helpers ───────────────────────────────────────────────────────────────────

def _module(node_id: str, name: str, file_path: str, mod_type: str = "source") -> NodeData:
    return NodeData(label="Module", id=node_id,
                    properties={"name": name, "file_path": file_path, "type": mod_type})


def _stub(name: str) -> NodeData:
    return NodeData(label="Module", id=name,
                    properties={"name": name, "file_path": "", "type": "external"})


def _class(node_id: str, name: str, file_path: str) -> NodeData:
    return NodeData(label="Class", id=node_id,
                    properties={"name": name, "file_path": file_path})


def _edge(src: str, rel: str, dst: str) -> EdgeData:
    return EdgeData(src_id=src, rel=rel, dst_id=dst)


# ── Module stub resolution (M1 behaviour preserved) ──────────────────────────

class TestModuleStubResolution:
    def test_stub_replaced_by_real_module(self):
        result = ParseResult(
            nodes=[
                _module("math_utils.py", "math_utils", "math_utils.py"),
                _stub("math_utils"),
            ],
            edges=[_edge("string_helpers.py", "IMPORTS", "math_utils")],
        )
        resolved = resolve(result)
        assert not any(n.id == "math_utils" for n in resolved.nodes), "Stub should be removed"
        import_edge = next(e for e in resolved.edges if e.rel == "IMPORTS")
        assert import_edge.dst_id == "math_utils.py"

    def test_unresolvable_stub_kept(self):
        result = ParseResult(
            nodes=[_stub("requests")],
            edges=[_edge("app.py", "IMPORTS", "requests")],
        )
        resolved = resolve(result)
        assert any(n.id == "requests" for n in resolved.nodes)
        assert resolved.edges[0].dst_id == "requests"

    def test_no_stubs_returns_same_object(self):
        result = ParseResult(
            nodes=[_module("a.py", "a", "a.py")],
            edges=[],
        )
        resolved = resolve(result)
        assert resolved is result  # fast path: same object

    def test_original_not_mutated(self):
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _stub("a"),
            ],
            edges=[_edge("b.py", "IMPORTS", "a")],
        )
        original_node_count = len(result.nodes)
        resolve(result)
        assert len(result.nodes) == original_node_count


# ── Cross-file class linking (M2) ─────────────────────────────────────────────

class TestCrossFileClassLinking:
    def test_inherits_stub_resolved_to_real_class(self):
        """
        file_b.py::Bar INHERITS stub 'Foo'
        file_a.py::Foo is a real Class in the same ParseResult
        → edge should be rewritten to file_a.py::Foo
        """
        result = ParseResult(
            nodes=[
                _class("file_a.py::Foo", "Foo", "file_a.py"),
                _class("file_b.py::Bar", "Bar", "file_b.py"),
                _stub("Foo"),  # created by parser _ensure_stub
            ],
            edges=[_edge("file_b.py::Bar", "INHERITS", "Foo")],
        )
        resolved = resolve(result)

        # Stub 'Foo' should be removed
        assert not any(n.id == "Foo" for n in resolved.nodes)
        # INHERITS edge now points to the real class
        inherit_edge = next(e for e in resolved.edges if e.rel == "INHERITS")
        assert inherit_edge.dst_id == "file_a.py::Foo"

    def test_unresolvable_class_stub_kept(self):
        """External ABC class stub should remain if not found in the repo."""
        result = ParseResult(
            nodes=[
                _class("mymod.py::Foo", "Foo", "mymod.py"),
                _stub("ABC"),
            ],
            edges=[_edge("mymod.py::Foo", "INHERITS", "ABC")],
        )
        resolved = resolve(result)
        assert any(n.id == "ABC" for n in resolved.nodes)
        assert resolved.edges[0].dst_id == "ABC"

    def test_class_stub_does_not_override_module_with_same_name(self):
        """
        If a real Module and a real Class both exist with the same simple name,
        the Module stub should resolve to the Module (not overridden by Class).
        """
        result = ParseResult(
            nodes=[
                _module("utils.py", "utils", "utils.py"),
                _class("models.py::Utils", "Utils", "models.py"),
                _stub("utils"),
            ],
            edges=[_edge("app.py", "IMPORTS", "utils")],
        )
        resolved = resolve(result)
        import_edge = next(e for e in resolved.edges if e.rel == "IMPORTS")
        # Should resolve to the Module, not the Class
        assert import_edge.dst_id == "utils.py"

    def test_multiple_stubs_resolved_independently(self):
        result = ParseResult(
            nodes=[
                _class("a.py::Foo", "Foo", "a.py"),
                _class("b.py::Bar", "Bar", "b.py"),
                _stub("Foo"),
                _stub("Bar"),
            ],
            edges=[
                _edge("c.py::Baz", "INHERITS", "Foo"),
                _edge("c.py::Baz", "INHERITS", "Bar"),
            ],
        )
        resolved = resolve(result)
        dst_ids = {e.dst_id for e in resolved.edges}
        assert "a.py::Foo" in dst_ids
        assert "b.py::Bar" in dst_ids
        assert "Foo" not in dst_ids
        assert "Bar" not in dst_ids


# ── stub_count helper ─────────────────────────────────────────────────────────

class TestStubCount:
    def test_counts_external_module_stubs(self):
        result = ParseResult(nodes=[_stub("a"), _stub("b"), _module("c.py", "c", "c.py")])
        assert stub_count(result) == 2

    def test_zero_when_no_stubs(self):
        result = ParseResult(nodes=[_module("a.py", "a", "a.py")])
        assert stub_count(result) == 0
