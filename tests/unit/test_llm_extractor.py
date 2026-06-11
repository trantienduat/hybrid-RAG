"""
Unit tests for ingestion/ollama_llm_extractor.py.
All HTTP calls are mocked — no Ollama service required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from hybrid_rag.ingestion.ollama_llm_extractor import OllamaLLMExtractor
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult

# ── helpers ───────────────────────────────────────────────────────────────────


def _parse_result(*node_ids: str) -> ParseResult:
    """Build a minimal ParseResult with Function/Class nodes for given IDs."""
    nodes = []
    for nid in node_ids:
        label = "Function" if "::" in nid else "Module"
        nodes.append(
            NodeData(
                label=label,
                id=nid,
                properties={"name": nid.split("::")[-1], "file_path": "a.py"},
            )
        )
    return ParseResult(nodes=nodes, edges=[])


def _mock_response(edges: list[dict]) -> MagicMock:
    """Return a mock httpx.Response yielding JSON with an edges list."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": json.dumps({"edges": edges})}
    return mock_resp


def _extractor() -> OllamaLLMExtractor:
    return OllamaLLMExtractor(ollama_url="http://localhost:11434", model="test-model")


# ── extract tests ─────────────────────────────────────────────────────────────


class TestOllamaLLMExtractor:
    def test_returns_empty_when_no_local_nodes(self):
        extractor = _extractor()
        # ParseResult with only external stubs
        result = ParseResult(
            nodes=[
                NodeData(
                    label="Module",
                    id="requests",
                    properties={"name": "requests", "file_path": "", "type": "external"},
                )
            ],
            edges=[],
        )
        edges = extractor.extract("import requests", result)
        assert edges == []

    def test_valid_response_produces_edge(self):
        extractor = _extractor()
        result = _parse_result("a.py::MyClass::my_method")
        # Add a Class node as destination
        result.nodes.append(
            NodeData(
                label="Class",
                id="b.py::SomeClass",
                properties={"name": "SomeClass", "file_path": "b.py"},
            )
        )

        llm_edges = [
            {
                "src_id": "a.py::MyClass::my_method",
                "rel": "USES",
                "dst_name": "SomeClass",
                "confidence": 0.9,
            }
        ]

        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("def my_method(self, x: SomeClass): ...", result)

        assert len(edges) == 1
        assert edges[0].src_id == "a.py::MyClass::my_method"
        assert edges[0].rel == "USES"
        assert edges[0].dst_id == "b.py::SomeClass"

    def test_low_confidence_edge_filtered(self):
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        llm_edges = [
            {"src_id": "a.py::foo", "rel": "USES", "dst_name": "SomeClass", "confidence": 0.5}
        ]  # below 0.7

        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("code", result)

        assert edges == []

    def test_builtin_type_skipped(self):
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        llm_edges = [
            {"src_id": "a.py::foo", "rel": "USES", "dst_name": "str", "confidence": 0.95},
            {"src_id": "a.py::foo", "rel": "USES", "dst_name": "Optional", "confidence": 0.95},
            {"src_id": "a.py::foo", "rel": "USES", "dst_name": "dict", "confidence": 0.95},
        ]
        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_generic_type_annotation_skipped(self):
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        llm_edges = [
            {"src_id": "a.py::foo", "rel": "USES", "dst_name": "List[Node]", "confidence": 0.95}
        ]
        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_unknown_src_id_skipped(self):
        extractor = _extractor()
        result = _parse_result("a.py::real_fn")
        llm_edges = [
            {"src_id": "nonexistent::fn", "rel": "USES", "dst_name": "SomeClass", "confidence": 0.9}
        ]
        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_invalid_json_returns_empty(self):
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "this is not json {{{"}
        with patch.object(extractor._client, "post", return_value=mock_resp):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_network_error_returns_empty(self):
        import httpx

        extractor = _extractor()
        result = _parse_result("a.py::foo")
        with patch.object(extractor._client, "post", side_effect=httpx.ConnectError("refused")):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_empty_edges_in_response(self):
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        with patch.object(extractor._client, "post", return_value=_mock_response([])):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_duplicate_edge_not_added_if_already_in_result(self):
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        result.nodes.append(
            NodeData(label="Class", id="b.py::Bar", properties={"name": "Bar", "file_path": "b.py"})
        )
        # Pre-existing edge
        result.edges.append(EdgeData(src_id="a.py::foo", rel="USES", dst_id="b.py::Bar"))
        llm_edges = [{"src_id": "a.py::foo", "rel": "USES", "dst_name": "Bar", "confidence": 0.9}]
        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("code", result)
        assert edges == []

    def test_unknown_dst_becomes_stub_id(self):
        """dst_name not in result → dst_id is set to the raw name for resolver to handle."""
        extractor = _extractor()
        result = _parse_result("a.py::foo")
        llm_edges = [
            {"src_id": "a.py::foo", "rel": "USES", "dst_name": "CrossFileClass", "confidence": 0.9}
        ]
        with patch.object(extractor._client, "post", return_value=_mock_response(llm_edges)):
            edges = extractor.extract("code", result)
        assert len(edges) == 1
        assert edges[0].dst_id == "CrossFileClass"

    def test_context_manager_calls_close(self):
        extractor = _extractor()
        with patch.object(extractor._client, "close") as mock_close:
            with extractor:
                pass
        mock_close.assert_called_once()
