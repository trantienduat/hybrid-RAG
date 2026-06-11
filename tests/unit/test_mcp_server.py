"""
Unit tests for the Model Context Protocol (MCP) server.

No external services required — all database stores and retrievers are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hybrid_rag.mcp.server import (
    get_ast_neighbors,
    get_community_report,
    health_check,
    list_repositories,
    query_codebase,
    search_ast_nodes,
)
from hybrid_rag.retrieval.context_assembler import RetrievalContext


class TestMCPServer:
    @patch("hybrid_rag.mcp.server.get_components")
    def test_query_codebase(self, mock_get_components):
        mock_retriever = MagicMock()
        mock_ctx = RetrievalContext(
            text="def hello():\n    print('world')",
            chunks=[],
            metadata={"total_results": 1, "shown": 1},
        )
        mock_retriever.retrieve_with_context.return_value = mock_ctx
        mock_get_components.return_value = (MagicMock(), MagicMock(), mock_retriever)

        # Call the tool
        res = query_codebase(question="how to print hello world?")

        # Assertions
        assert "def hello():" in res
        mock_retriever.retrieve_with_context.assert_called_once_with(
            query="how to print hello world?",
            top_k=20,
            max_tokens=2048,
            repository=None,
        )

    @patch("hybrid_rag.mcp.server.get_components")
    def test_query_codebase_empty(self, mock_get_components):
        mock_retriever = MagicMock()
        mock_ctx = RetrievalContext(
            text="",
            chunks=[],
            metadata={"total_results": 0, "shown": 0},
        )
        mock_retriever.retrieve_with_context.return_value = mock_ctx
        mock_get_components.return_value = (MagicMock(), MagicMock(), mock_retriever)

        res = query_codebase(question="does not exist")
        assert "No relevant codebase context" in res

    @patch("hybrid_rag.mcp.server.get_components")
    def test_list_repositories(self, mock_get_components):
        mock_graph_store = MagicMock()
        mock_graph_store.list_repositories.return_value = ["repo-one", "repo-two"]
        mock_get_components.return_value = (mock_graph_store, MagicMock(), MagicMock())

        repos = list_repositories()
        assert repos == ["repo-one", "repo-two"]
        mock_graph_store.list_repositories.assert_called_once()

    @patch("hybrid_rag.mcp.server.get_components")
    def test_search_ast_nodes(self, mock_get_components):
        mock_graph_store = MagicMock()
        mock_nodes = [
            {
                "node_id": "package.module.Class",
                "label": "Class",
                "name": "Class",
                "file_path": "package/module.py",
                "repository": "repo-one",
            }
        ]
        mock_graph_store.find_nodes.return_value = mock_nodes
        mock_get_components.return_value = (mock_graph_store, MagicMock(), MagicMock())

        nodes = search_ast_nodes(query="Class", repository="repo-one", limit=10)
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Class"
        mock_graph_store.find_nodes.assert_called_once_with(
            name="Class", repository="repo-one", limit=10
        )

    @patch("hybrid_rag.mcp.server.get_components")
    def test_get_ast_neighbors(self, mock_get_components):
        mock_graph_store = MagicMock()
        mock_graph_store.find_neighbors.return_value = [
            {
                "rel": "CALLS",
                "dst_id": "package.module.Class.other_method",
                "dst_name": "other_method",
                "dst_label": "Function",
                "dst_file_path": "package/module.py",
                "dst_repository": "repo-one",
            }
        ]
        mock_get_components.return_value = (mock_graph_store, MagicMock(), MagicMock())

        edges = get_ast_neighbors(node_id="package.module.Class.method", direction="out", limit=5)
        assert len(edges) == 1
        assert edges[0]["rel"] == "CALLS"
        mock_graph_store.find_neighbors.assert_called_once_with(
            node_id="package.module.Class.method", direction="out", max_hops=1, limit=5
        )

    @patch("hybrid_rag.mcp.server.get_components")
    def test_get_community_report(self, mock_get_components):
        mock_graph_store = MagicMock()
        mock_query_res = MagicMock()
        mock_query_res.result_set = [["community_1", "Auth System", "Manages user registration", 0]]
        mock_graph_store.query.return_value = mock_query_res
        mock_get_components.return_value = (mock_graph_store, MagicMock(), MagicMock())

        communities = get_community_report()
        assert len(communities) == 1
        assert communities[0]["name"] == "Auth System"
        assert communities[0]["level"] == 0
        mock_graph_store.query.assert_called_once()

    async def test_health_check(self):
        from starlette.datastructures import Headers
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": Headers().raw,
        }
        mock_request = Request(scope)

        resp = await health_check(mock_request)
        assert resp.status_code == 200
        assert b'"status":"ok"' in resp.body
