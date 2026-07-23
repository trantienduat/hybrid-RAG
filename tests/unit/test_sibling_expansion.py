from unittest.mock import MagicMock

from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever


def test_get_called_siblings():
    graph_store = MagicMock()
    mock_res = MagicMock()
    mock_res.result_set = [["sibling_method"]]
    graph_store.query.return_value = mock_res

    retriever = HybridRetriever(
        graph_store=graph_store, vector_store=MagicMock(), embedder=MagicMock()
    )

    siblings = retriever._get_called_siblings("module.Class.method")
    assert siblings == ["sibling_method"]
    graph_store.query.assert_called_once()
    cypher_called = graph_store.query.call_args[0][0]
    assert "CALLS" in cypher_called


def test_retrieve_with_context_attaches_siblings(monkeypatch):
    graph_store = MagicMock()
    mock_res = MagicMock()
    mock_res.result_set = [["sibling_method"]]
    graph_store.query.return_value = mock_res

    retriever = HybridRetriever(
        graph_store=graph_store, vector_store=MagicMock(), embedder=MagicMock()
    )

    # Mock retrieve to return a single function result
    monkeypatch.setattr(
        retriever,
        "retrieve",
        lambda query, top_k, repository: [
            {
                "node_id": "module.Class.method",
                "name": "method",
                "label": "Function",
                "file_path": "src/module.py",
                "repository": "my_repo",
                "source": "graph",
            }
        ],
    )

    # Mock ContextAssembler assemble method
    mock_assembler = MagicMock()
    retriever._assembler = mock_assembler

    retriever.retrieve_with_context("query")

    args, kwargs = mock_assembler.assemble.call_args
    passed_results = args[0]
    assert len(passed_results) == 1
    assert passed_results[0]["called_siblings"] == ["sibling_method"]
