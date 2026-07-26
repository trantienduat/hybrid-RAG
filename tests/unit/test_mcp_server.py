"""
Unit tests for the Model Context Protocol (MCP) server.

No external services required — all database stores and retrievers are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from hybrid_rag.mcp.server import (
    _index_status,
    get_ast_neighbors,
    get_community_report,
    get_index_status,
    health_check,
    list_repositories,
    query_codebase,
    search_ast_nodes,
)
from hybrid_rag.retrieval.context_assembler import RetrievalContext


class TestMCPServer:
    @patch("hybrid_rag.mcp.server.app_config.get_repo_path", return_value=None)
    def test_legacy_index_metadata_is_stale(self, _mock_repo_path):
        graph_store = MagicMock()
        graph_store.get_repository_metadata.return_value = {"last_indexed_commit": "wrong-commit"}
        vector_store = MagicMock()
        vector_store.get_repository_metadata.return_value = {
            "source_identity": set(),
            "index_run_id": set(),
            "indexed_commit": {"wrong-commit"},
        }

        status = _index_status(graph_store, vector_store, "legacy-repo")

        assert status["stale"] is True
        assert status["provenance_valid"] is False
        assert "missing source_identity" in status["provenance_error"]

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
            max_tokens=1024,
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

    @patch("hybrid_rag.mcp.server.get_components")
    def test_get_community_report_scopes_repository(self, mock_get_components):
        mock_graph_store = MagicMock()
        mock_graph_store.query.return_value.result_set = []
        mock_get_components.return_value = (mock_graph_store, MagicMock(), MagicMock())

        get_community_report(repository="repo-one")

        cypher, params = mock_graph_store.query.call_args.args
        assert "n.repository = $repository" in cypher
        assert params == {"repository": "repo-one"}

    @patch("hybrid_rag.mcp.server._index_status")
    @patch("hybrid_rag.mcp.server.get_components")
    def test_get_index_status(self, mock_get_components, mock_status):
        mock_status.return_value = {
            "repository": "repo-one",
            "indexed_commit": "abc",
            "current_commit": "def",
            "working_tree_dirty": False,
            "stale": True,
            "updated_at": 123,
        }
        graph_store = MagicMock()
        vector_store = MagicMock()
        mock_get_components.return_value = (graph_store, vector_store, MagicMock())

        status = get_index_status("repo-one")

        assert status["stale"] is True
        mock_status.assert_called_once_with(graph_store, vector_store, "repo-one")

    @patch("hybrid_rag.mcp.server.get_components")
    def test_tool_failure_is_explicit(self, mock_get_components):
        mock_get_components.side_effect = RuntimeError("database unavailable")

        with pytest.raises(ToolError):
            list_repositories()

    @patch("hybrid_rag.mcp.server.httpx.AsyncClient")
    @patch("hybrid_rag.mcp.server.get_components")
    async def test_health_check(self, mock_get_components, mock_client_class):
        graph_store = MagicMock()
        graph_store.node_count.return_value = 10
        vector_store = MagicMock()
        vector_store.point_count.return_value = 20
        mock_get_components.return_value = (graph_store, vector_store, MagicMock())
        response = MagicMock()
        response.raise_for_status.return_value = None
        async_client = AsyncMock()
        async_client.get.return_value = response
        mock_client_class.return_value.__aenter__.return_value = async_client

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
        assert b'"falkordb"' in resp.body
        assert b'"qdrant"' in resp.body
        assert b'"ollama"' in resp.body
