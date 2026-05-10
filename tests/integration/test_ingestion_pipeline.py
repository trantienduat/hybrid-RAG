"""
Integration test: full M1 ingestion pipeline.

Requires live services:
  - FalkorDB on localhost:6379
  - Qdrant on localhost:6333
  - Ollama on localhost:11434 with nomic-embed-text

Run: pytest tests/integration/test_ingestion_pipeline.py -v -s
"""
from __future__ import annotations

import pytest
from pathlib import Path

# ── Skip if services unavailable ──────────────────────────────────

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


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graph_client():
    from hybrid_rag.graph.client import GraphClient
    client = GraphClient(graph_name="test_codebase")
    client.clear()
    yield client
    client.clear()


@pytest.fixture(scope="module")
def vector_client():
    from hybrid_rag.vector.client import VectorClient
    client = VectorClient(collection="test_code_chunks")
    client.clear()
    yield client
    client.clear()


@pytest.fixture(scope="module")
def embedder():
    from hybrid_rag.ingestion.embedder import Embedder
    with Embedder() as emb:
        yield emb


# ── Helpers ───────────────────────────────────────────────────────

def _fixture_file() -> Path:
    """Return path to parser.py as a small real-world fixture."""
    return Path("src/hybrid_rag/ingestion/parser.py")


# ── Tests ─────────────────────────────────────────────────────────

@requires_services
class TestIngestionPipeline:

    def test_parse_produces_nodes_and_edges(self):
        from hybrid_rag.ingestion.parser import parse_file
        result = parse_file(_fixture_file(), Path("."))
        assert len(result.nodes) > 5, "Expected multiple nodes"
        assert len(result.edges) > 5, "Expected multiple edges"
        assert result.errors == [], f"Unexpected errors: {result.errors}"

    def test_triplet_extraction(self):
        from hybrid_rag.ingestion.parser import parse_file
        from hybrid_rag.ingestion.triplet_extractor import extract_triples
        result = parse_file(_fixture_file(), Path("."))
        triples = extract_triples(result)
        assert len(triples) > 0
        rels = {t.rel for t in triples}
        assert "DEFINES" in rels
        assert "DEFINED_IN" in rels

    def test_graph_ingest(self, graph_client):
        from hybrid_rag.ingestion.parser import parse_file
        result = parse_file(_fixture_file(), Path("."))
        counts = graph_client.ingest(result)
        assert counts["nodes"] > 0
        assert graph_client.node_count() > 0

    def test_embed_nodes(self, embedder):
        from hybrid_rag.ingestion.parser import parse_file
        result = parse_file(_fixture_file(), Path("."))
        chunks = embedder.embed_nodes(result.nodes)
        assert len(chunks) > 0
        # Check embedding dimension
        assert len(chunks[0]["embedding"]) == 768
        # Check all required keys
        for chunk in chunks:
            assert "node_id" in chunk
            assert "text" in chunk
            assert "embedding" in chunk

    def test_vector_upsert(self, vector_client, embedder):
        from hybrid_rag.ingestion.parser import parse_file
        result = parse_file(_fixture_file(), Path("."))
        chunks = embedder.embed_nodes(result.nodes)
        count = vector_client.upsert(chunks)
        assert count > 0
        assert vector_client.point_count() > 0

    def test_vector_search(self, vector_client, embedder):
        query = "function that parses a file using tree-sitter"
        embedding = embedder.embed_query(query)
        hits = vector_client.search(embedding, top_k=5)
        assert len(hits) > 0
        assert "node_id" in hits[0]
        assert "score" in hits[0]
        assert hits[0]["score"] > 0.0

    def test_full_pipeline_single_file(self, graph_client, vector_client, embedder):
        """End-to-end: parse → graph → embed → vector."""
        from hybrid_rag.ingestion.parser import parse_file
        result = parse_file(_fixture_file(), Path("."))

        g_counts = graph_client.ingest(result)
        chunks = embedder.embed_nodes(result.nodes)
        v_count = vector_client.upsert(chunks)

        assert g_counts["nodes"] > 0
        assert v_count > 0

        # Cross-check: a node from graph should be findable by vector search
        query_fn = next(n for n in result.nodes if n.label == "Function")
        q_embed = embedder.embed_query(query_fn.properties.get("signature", query_fn.id))
        hits = vector_client.search(q_embed, top_k=10)
        hit_ids = {h["node_id"] for h in hits}
        assert query_fn.id in hit_ids, f"{query_fn.id} not found in top-10 vector results"
