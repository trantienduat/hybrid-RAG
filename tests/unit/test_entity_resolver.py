"""
Unit tests for ingestion/entity_resolver.py — including M2 cross-file class linking.
No external services required.
"""

from __future__ import annotations

from hybrid_rag.ingestion.entity_resolver import resolve, stub_count
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult

# ── helpers ───────────────────────────────────────────────────────────────────


def _module(node_id: str, name: str, file_path: str, mod_type: str = "source") -> NodeData:
    return NodeData(
        label="Module",
        id=node_id,
        properties={"name": name, "file_path": file_path, "type": mod_type},
    )


def _stub(name: str) -> NodeData:
    return NodeData(
        label="Module", id=name, properties={"name": name, "file_path": "", "type": "external"}
    )


def _class(node_id: str, name: str, file_path: str) -> NodeData:
    return NodeData(label="Class", id=node_id, properties={"name": name, "file_path": file_path})


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


def _function(node_id: str, name: str, file_path: str) -> NodeData:
    return NodeData(label="Function", id=node_id, properties={"name": name, "file_path": file_path})


def _call_stub(name: str) -> NodeData:
    return NodeData(
        label="Function",
        id=f"__call__{name}",
        properties={"name": name, "file_path": "", "type": "external"},
    )


# ── AST Call Stub Resolution ──────────────────────────────────────────────────


class TestASTCallResolver:
    def test_resolve_sibling_class_method(self):
        """__call__helper inside class context should resolve to sibling method."""
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _class("a.py::MyClass", "MyClass", "a.py"),
                _function("a.py::MyClass.helper", "helper", "a.py"),
                _function("a.py::MyClass.main", "main", "a.py"),
            ],
            edges=[
                _edge("a.py::MyClass.main", "CALLS", "__call__helper"),
            ],
        )
        resolved = resolve(result)
        call_edge = resolved.edges[0]
        assert call_edge.dst_id == "a.py::MyClass.helper"

    def test_resolve_same_module_function(self):
        """__call__helper should resolve to same-module function if class context is not sibling."""
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _function("a.py.helper", "helper", "a.py"),
                _function("a.py.main", "main", "a.py"),
            ],
            edges=[
                _edge("a.py.main", "CALLS", "__call__helper"),
            ],
        )
        resolved = resolve(result)
        call_edge = resolved.edges[0]
        assert call_edge.dst_id == "a.py.helper"

    def test_global_unique_name_remains_unresolved(self):
        """A unique simple name is insufficient evidence for cross-module resolution."""
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _function("a.py.main", "main", "a.py"),
                _module("b.py", "b", "b.py"),
                _function("b.py.unique_func", "unique_func", "b.py"),
                _call_stub("unique_func"),
            ],
            edges=[
                EdgeData(
                    src_id="a.py.main",
                    rel="CALLS",
                    dst_id="__call__unique_func",
                    properties={"callee_expr": "unique_func"},
                ),
            ],
        )
        resolved = resolve(result)
        call_edge = resolved.edges[0]
        assert call_edge.dst_id == "__call__unique_func"
        assert any(node.id == "__call__unique_func" for node in resolved.nodes)

    def test_resolve_imported_function(self):
        """An explicitly imported function should resolve through Tier C."""
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _function("a.py.main", "main", "a.py"),
                _module("b.py", "b", "b.py"),
                _function("b.py.helper", "helper", "b.py"),
                _call_stub("helper"),
            ],
            edges=[
                _edge("a.py", "IMPORTS", "helper"),
                EdgeData(
                    src_id="a.py.main",
                    rel="CALLS",
                    dst_id="__call__helper",
                    properties={"callee_expr": "helper"},
                ),
            ],
        )
        resolved = resolve(result)
        call_edge = next(edge for edge in resolved.edges if edge.rel == "CALLS")
        assert call_edge.dst_id == "b.py.helper"

    def test_receiver_qualified_call_does_not_resolve_by_simple_name(self):
        """A receiver-qualified call must not bind to an unrelated same-name function."""
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _function("a.py.add", "add", "a.py"),
                _function("a.py.main", "main", "a.py"),
                _call_stub("add"),
            ],
            edges=[
                EdgeData(
                    src_id="a.py.main",
                    rel="CALLS",
                    dst_id="__call__add",
                    properties={"callee_expr": "other.add"},
                ),
            ],
        )
        resolved = resolve(result)
        assert resolved.edges[0].dst_id == "__call__add"

    def test_call_resolution_is_edge_local(self):
        """Resolving one call stub must not rewrite an ambiguous same-name call."""
        result = ParseResult(
            nodes=[
                _module("a.py", "a", "a.py"),
                _class("a.py::Calculator", "Calculator", "a.py"),
                _function("a.py::Calculator.add", "add", "a.py"),
                _function("a.py::Calculator.run", "run", "a.py"),
                _function("a.py.main", "main", "a.py"),
                _call_stub("add"),
            ],
            edges=[
                EdgeData(
                    src_id="a.py::Calculator.run",
                    rel="CALLS",
                    dst_id="__call__add",
                    properties={"callee_expr": "self.add"},
                ),
                EdgeData(
                    src_id="a.py.main",
                    rel="CALLS",
                    dst_id="__call__add",
                    properties={"callee_expr": "other.add"},
                ),
            ],
        )
        resolved = resolve(result)
        assert resolved.edges[0].dst_id == "a.py::Calculator.add"
        assert resolved.edges[1].dst_id == "__call__add"
        assert any(node.id == "__call__add" for node in resolved.nodes)

    def test_stub_count_includes_all_stubs(self):
        """stub_count should count external module, class, and __call__ stubs."""
        result = ParseResult(
            nodes=[
                _stub("mod_stub"),
                NodeData(label="Class", id="ClassStub", properties={"type": "external"}),
                NodeData(label="Function", id="__call__func", properties={"type": "external"}),
            ],
            edges=[],
        )
        assert stub_count(result) == 3

    def test_resolve_global_batched_query(self):
        """resolve_global should execute a single batched Cypher query."""
        from unittest.mock import MagicMock

        from hybrid_rag.ingestion.entity_resolver import resolve_global

        result = ParseResult(
            nodes=[
                _stub("ext_module"),
                NodeData(label="Class", id="ExtClass", properties={"type": "external"}),
            ],
            edges=[],
        )

        mock_graph_store = MagicMock()
        mock_result = MagicMock()
        mock_result.result_set = [["ext_module", "real.ext_module"]]
        mock_graph_store.query.return_value = mock_result

        resolved = resolve_global(result, mock_graph_store)

        assert mock_graph_store.query.call_count == 1
        query_args = mock_graph_store.query.call_args[0]
        assert "UNWIND $stubs AS stub" in query_args[0]
        assert not any(n.id == "ext_module" for n in resolved.nodes)
