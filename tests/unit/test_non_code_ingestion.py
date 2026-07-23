from unittest.mock import MagicMock

from hybrid_rag.ingestion.chunker import chunk_file
from hybrid_rag.ingestion.parser import parse_file
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever


def test_parse_non_code_file(tmp_path):
    yaml_file = tmp_path / "docker-compose.yml"
    yaml_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx", encoding="utf-8")

    res = parse_file(yaml_file, tmp_path, repo_name="test_repo")
    assert len(res.nodes) == 1
    assert res.nodes[0].label == "Configuration"
    assert res.nodes[0].id == "test_repo::docker-compose.yml"
    assert res.nodes[0].properties["name"] == "docker-compose.yml"
    assert "nginx" in res.nodes[0].properties["text"]

    assert len(res.edges) == 1
    assert res.edges[0].rel == "WEAK_LINK"
    assert res.edges[0].src_id == "test_repo::docker-compose.yml"
    assert res.edges[0].dst_id == "test_repo"


def test_chunk_non_code_file(tmp_path):
    yaml_file = tmp_path / "docker-compose.yml"
    yaml_file.write_text("version: '3'\nservices:\n  web:\n    image: nginx", encoding="utf-8")

    chunks = chunk_file(yaml_file, tmp_path)
    assert len(chunks) == 1
    assert chunks[0].label == "Configuration"
    assert "nginx" in chunks[0].text
    assert "docker-compose.yml" in chunks[0].text


def test_query_routing_activation(monkeypatch):
    monkeypatch.setenv("QUERY_ROUTING", "true")

    graph_store = MagicMock()
    vector_store = MagicMock()
    embedder = MagicMock()

    retriever = HybridRetriever(
        graph_store=graph_store, vector_store=vector_store, embedder=embedder
    )

    mock_retrieve = MagicMock(return_value=[])
    monkeypatch.setattr(retriever._vector_retriever, "retrieve", mock_retrieve)

    # 1. Ask code query
    retriever.retrieve("how does function add work?")
    args, kwargs = mock_retrieve.call_args
    assert kwargs["filter_payload"]["file_type"] == "code"

    # 2. Ask config query
    retriever.retrieve("what port does docker compose use?")
    args, kwargs = mock_retrieve.call_args
    assert kwargs["filter_payload"]["file_type"] == "non_code"
