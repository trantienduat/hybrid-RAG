"""
Integration tests for the M3 hybrid retrieval pipeline.

Requires live services:
  - FalkorDB on localhost:6379
  - Qdrant on localhost:6333
  - Ollama on localhost:11434 with nomic-embed-text

These tests index the small fixture repo, run retrieval queries, and assert
that results have the correct shape and sources.

Run: pytest tests/integration/test_retrieval_pipeline.py -v -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Service availability helpers ──────────────────────────────────


def _falkordb_ok() -> bool:
    try:
        import falkordb

        db = falkordb.FalkorDB(host="localhost", port=6379)
        db.connection.ping()
        return True
    except Exception:
        return False


def _qdrant_ok() -> bool:
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(host="localhost", port=6333)
        c.get_collections()
        return True
    except Exception:
        return False


def _ollama_ok() -> bool:
    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.integration

requires_services = pytest.mark.skipif(
    not (_falkordb_ok() and _qdrant_ok() and _ollama_ok()),
    reason="FalkorDB / Qdrant / Ollama not reachable",
)

_FIXTURE_REPO = Path(__file__).parent.parent.parent / "fixtures" / "small_repo"
_GRAPH_NAME = "test_retrieval_m3"
_COLLECTION = "test_retrieval_m3"


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def indexed_retriever():
    """Index the small fixture repo and return a ready HybridRetriever."""
    from hybrid_rag.graph.falkordb_store import FalkorDBStore
    from hybrid_rag.ingestion.chunker import chunk_file
    from hybrid_rag.ingestion.entity_resolver import resolve
    from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
    from hybrid_rag.ingestion.parser import parse_repo
    from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
    from hybrid_rag.vector.qdrant_store import QdrantStore

    graph_store = FalkorDBStore(graph_name=_GRAPH_NAME)
    graph_store.clear()

    vector_store = QdrantStore(collection=_COLLECTION)
    vector_store.clear()

    embedder = OllamaEmbedder()

    # Index fixture repo
    result = parse_repo(_FIXTURE_REPO, languages=["python"])
    result = resolve(result)
    graph_store.ingest(result)

    all_chunks = []
    seen_files: set[str] = set()
    for node in result.nodes:
        fp = node.properties.get("file_path", "")
        if fp and fp not in seen_files:
            seen_files.add(fp)
            for ch in chunk_file(_FIXTURE_REPO / fp, _FIXTURE_REPO):
                emb = embedder.embed_query(ch.text)
                all_chunks.append(
                    {
                        "node_id": f"{ch.node_id}::{ch.chunk_index}",
                        "label": ch.label,
                        "file_path": ch.file_path,
                        "text": ch.text,
                        "embedding": emb,
                    }
                )
    vector_store.upsert(all_chunks)

    retriever = HybridRetriever(
        graph_store=graph_store,
        vector_store=vector_store,
        embedder=embedder,
    )
    yield retriever

    # Teardown
    retriever.close()
    graph_store.clear()
    vector_store.clear()


# ── Tests ─────────────────────────────────────────────────────────


@requires_services
class TestHybridRetrieverIntegration:
    def test_retrieve_returns_results(self, indexed_retriever):
        results = indexed_retriever.retrieve("what functions are defined?", top_k=5)
        assert len(results) > 0

    def test_results_have_required_keys(self, indexed_retriever):
        results = indexed_retriever.retrieve("show me math functions", top_k=3)
        for r in results:
            assert "node_id" in r
            assert "rrf_score" in r
            assert "source" in r

    def test_retrieve_with_context_has_text(self, indexed_retriever):
        ctx = indexed_retriever.retrieve_with_context("explain the math utilities", top_k=5)
        assert ctx.text
        assert ctx.metadata["total_results"] >= 0

    def test_structural_query_uses_graph(self, indexed_retriever):
        results = indexed_retriever.retrieve("which functions are defined in math_utils?", top_k=10)
        sources = {r.get("source") for r in results}
        assert sources & {"graph", "hybrid"}

    def test_semantic_query_returns_vector_results(self, indexed_retriever):
        results = indexed_retriever.retrieve("explain how string operations work", top_k=5)
        sources = {r.get("source") for r in results}
        assert sources & {"vector", "hybrid"}

    def test_rrf_scores_descending(self, indexed_retriever):
        results = indexed_retriever.retrieve("math operations", top_k=10)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_context_metadata_populated(self, indexed_retriever):
        ctx = indexed_retriever.retrieve_with_context("show me string helpers", top_k=5)
        assert "total_results" in ctx.metadata
        assert "shown" in ctx.metadata
        assert isinstance(ctx.metadata["has_graph"], bool)
        assert isinstance(ctx.metadata["has_vector"], bool)


@requires_services
class TestGraphStoreRetrievalMethods:
    """Integration tests for the new find_nodes / find_neighbors methods on FalkorDBStore."""

    @pytest.fixture(scope="class")
    def graph_store(self):
        from hybrid_rag.graph.falkordb_store import FalkorDBStore
        from hybrid_rag.ingestion.entity_resolver import resolve
        from hybrid_rag.ingestion.parser import parse_repo

        store = FalkorDBStore(graph_name=f"{_GRAPH_NAME}_store")
        store.clear()
        result = parse_repo(_FIXTURE_REPO, languages=["python"])
        result = resolve(result)
        store.ingest(result)
        yield store
        store.clear()

    def test_find_nodes_returns_list(self, graph_store):
        nodes = graph_store.find_nodes("add")
        assert isinstance(nodes, list)

    def test_find_nodes_has_required_keys(self, graph_store):
        nodes = graph_store.find_nodes("add")
        for n in nodes:
            assert "node_id" in n
            assert "label" in n
            assert "name" in n
            assert "file_path" in n

    def test_find_nodes_empty_name_returns_empty(self, graph_store):
        assert graph_store.find_nodes("") == []

    def test_find_nodes_nonexistent_returns_empty(self, graph_store):
        assert graph_store.find_nodes("ZZZ_definitely_not_a_real_name_XYZ") == []

    def test_find_neighbors_returns_list(self, graph_store):
        nodes = graph_store.find_nodes("add")
        if not nodes:
            pytest.skip("No 'add' nodes in fixture")
        nid = nodes[0]["node_id"]
        neighbors = graph_store.find_neighbors(nid, direction="both")
        assert isinstance(neighbors, list)

    def test_find_neighbors_has_required_keys(self, graph_store):
        nodes = graph_store.find_nodes("add")
        if not nodes:
            pytest.skip("No 'add' nodes in fixture")
        nid = nodes[0]["node_id"]
        neighbors = graph_store.find_neighbors(nid, direction="both")
        for nb in neighbors:
            assert "src_id" in nb
            assert "rel" in nb
            assert "dst_id" in nb
