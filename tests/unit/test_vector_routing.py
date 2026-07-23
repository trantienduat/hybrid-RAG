from unittest.mock import MagicMock

from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever


def test_vector_routing_active(monkeypatch):
    monkeypatch.setenv("VECTOR_ROUTING", "true")
    monkeypatch.setenv("ROUTING_FALLBACK", "false")
    monkeypatch.setenv("ROUTING_ANCHORS", "2")

    graph_store = MagicMock()
    graph_store.find_neighbors.return_value = [
        {
            "dst_id": "neighbor_node",
            "dst_name": "neighbor",
            "dst_label": "Function",
            "dst_file_path": "src/neighbor.py",
            "dst_repository": "my_repo",
            "rel": "CALLS",
        }
    ]

    vector_store = MagicMock()
    embedder = MagicMock()

    retriever = HybridRetriever(
        graph_store=graph_store, vector_store=vector_store, embedder=embedder
    )

    monkeypatch.setattr(
        retriever._vector_retriever,
        "retrieve",
        lambda query, top_k, filter_payload: [
            {
                "node_id": "anchor_node::0",
                "base_node_id": "anchor_node",
                "name": "anchor",
                "label": "Function",
                "file_path": "src/anchor.py",
                "repository": "my_repo",
                "text": "def anchor(): pass",
                "score": 0.85,
                "source": "vector",
            }
        ],
    )

    results = retriever.retrieve("test query")
    assert len(results) == 2
    assert results[0]["node_id"] == "anchor_node"
    assert results[1]["node_id"] == "neighbor_node"
    graph_store.find_neighbors.assert_any_call("anchor_node", direction="in", max_hops=1, limit=10)
    graph_store.find_neighbors.assert_any_call("anchor_node", direction="out", max_hops=1, limit=10)


def test_vector_routing_fallback(monkeypatch):
    monkeypatch.setenv("VECTOR_ROUTING", "true")
    monkeypatch.setenv("ROUTING_FALLBACK", "true")
    monkeypatch.setenv("ROUTING_THRESHOLD", "0.90")

    graph_store = MagicMock()
    vector_store = MagicMock()
    embedder = MagicMock()

    retriever = HybridRetriever(
        graph_store=graph_store, vector_store=vector_store, embedder=embedder
    )

    monkeypatch.setattr(
        retriever._vector_retriever,
        "retrieve",
        lambda query, top_k, filter_payload: [
            {
                "node_id": "anchor_node::0",
                "base_node_id": "anchor_node",
                "name": "anchor",
                "label": "Function",
                "file_path": "src/anchor.py",
                "repository": "my_repo",
                "text": "def anchor(): pass",
                "score": 0.80,
                "source": "vector",
            }
        ],
    )

    monkeypatch.setattr(
        retriever._graph_retriever,
        "retrieve",
        lambda analysis, top_k, repository: [
            {
                "node_id": "fallback_graph_node",
                "base_node_id": "fallback_graph_node",
                "name": "fallback",
                "label": "Function",
                "file_path": "src/fallback.py",
                "repository": "my_repo",
                "source": "graph",
            }
        ],
    )

    results = retriever.retrieve("test query")
    assert any(r["node_id"] == "fallback_graph_node" for r in results)
