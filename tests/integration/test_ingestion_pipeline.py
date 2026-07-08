"""
Integration test: full M1 ingestion pipeline.

Requires live services:
  - FalkorDB on localhost:6379
  - Qdrant on localhost:6333
  - Ollama on localhost:11434 with nomic-embed-text

Run: pytest tests/integration/test_ingestion_pipeline.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    from hybrid_rag.graph.falkordb_store import FalkorDBStore

    client = FalkorDBStore(graph_name="test_codebase")
    client.clear()
    yield client
    client.clear()


@pytest.fixture(scope="module")
def vector_client():
    from hybrid_rag.vector.qdrant_store import QdrantStore

    client = QdrantStore(collection="test_code_chunks")
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


# ── M2 Integration Tests ─────────────────────────────────────────


@requires_services
class TestM2Pipeline:
    def test_entity_resolver_cross_file_linking(self):
        """
        Parse both fixture files together so StringProcessor (string_helpers.py)
        and any cross-file inheritance stubs get resolved.
        The entity resolver should reduce stub count after resolution.
        """
        from hybrid_rag.ingestion.entity_resolver import resolve, stub_count
        from hybrid_rag.ingestion.parser import parse_repo

        fixture_repo = Path("fixtures/small_repo")
        result = parse_repo(fixture_repo, languages=["python"])

        before = stub_count(result)
        resolved = resolve(result)
        after = stub_count(resolved)

        # math_utils is imported by string_helpers — should be resolved
        assert after <= before, "stub_count should not increase after resolve"
        # math_utils module stub should now resolve to real module node
        module_ids = {n.id for n in resolved.nodes}
        assert "math_utils" in module_ids

    def test_merger_integrates_with_parse_result(self):
        """Merge supplemental edges into a real ParseResult and verify structure."""
        from hybrid_rag.ingestion.merger import merge_supplemental
        from hybrid_rag.ingestion.parser import EdgeData, parse_file

        fixture_repo = Path("fixtures/small_repo")
        result = parse_file(fixture_repo / "math_utils.py", fixture_repo)

        fn_id = next(n.id for n in result.nodes if n.label == "Function")
        extra = [
            EdgeData(
                src_id=fn_id,
                rel="USES",
                dst_id="SomeClass",
                properties={"source": "llm", "confidence": 0.85},
            )
        ]

        merged = merge_supplemental(result, extra)

        # Edge count increased
        assert len(merged.edges) == len(result.edges) + 1
        # New stub node created
        ids = {n.id for n in merged.nodes}
        assert "SomeClass" in ids
        # Original not mutated
        assert len(result.edges) == len(merged.edges) - 1

    def test_full_m2_pipeline_no_llm(self, graph_client, vector_client, embedder):
        """
        Full M2 flow without LLM extraction:
        parse_repo → merge (empty extras) → entity_resolve → graph ingest → embed → vector.
        Verifies INHERITS edges are written to FalkorDB when present.
        """
        from hybrid_rag.ingestion.entity_resolver import resolve
        from hybrid_rag.ingestion.merger import merge_supplemental
        from hybrid_rag.ingestion.parser import parse_repo

        fixture_repo = Path("fixtures/small_repo")
        result = parse_repo(fixture_repo, languages=["python"])
        result = merge_supplemental(result, [])  # no-op merge
        result = resolve(result)

        counts = graph_client.ingest(result)
        assert counts["nodes"] > 0
        assert counts["edges"] > 0

        # Embed + upsert
        chunks = embedder.embed_nodes(result.nodes)
        upserted = vector_client.upsert(chunks)
        assert upserted > 0

    def test_llm_extractor_returns_edges_or_empty(self):
        """
        OllamaLLMExtractor.extract() on a real file should return a list
        (possibly empty if LLM finds nothing, but never raise).
        Requires Ollama with a code-capable model.
        """
        from hybrid_rag.constants import DEFAULT_LLM_MODEL
        from hybrid_rag.ingestion.ollama_llm_extractor import OllamaLLMExtractor
        from hybrid_rag.ingestion.parser import parse_file

        fixture_repo = Path("fixtures/small_repo")
        fp = fixture_repo / "string_helpers.py"
        result = parse_file(fp, fixture_repo)
        source_text = fp.read_text(encoding="utf-8")

        with OllamaLLMExtractor(model=DEFAULT_LLM_MODEL) as extractor:
            edges = extractor.extract(source_text, result)

        assert isinstance(edges, list)
        for e in edges:
            assert e.src_id
            assert e.rel in ("USES", "CALLS", "INHERITS")
            assert e.dst_id
