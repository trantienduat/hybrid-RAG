import pytest

from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.entity_resolver import resolve, resolve_global
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.vector.qdrant_store import QdrantStore

# Helper local resolution alias
resolve_local = resolve


@pytest.fixture
def active_graph_store():
    # Attempt to connect to local active FalkorDB (active on localhost:6379)
    try:
        store = FalkorDBStore(host="localhost", port=6379, graph_name="test_cross_repo")
        store.clear()
        return store
    except Exception:
        pytest.skip("Local FalkorDB is not active on localhost:6379, skipping integration tests.")


@pytest.fixture
def active_vector_store():
    # Attempt to connect to local active Qdrant (active on localhost:6333)
    try:
        store = QdrantStore(host="localhost", port=6333, collection="test_cross_repo_chunks")
        store.clear()
        return store
    except Exception:
        pytest.skip("Local Qdrant is not active on localhost:6333, skipping integration tests.")


def test_resolve_global_links_dependent_repo_to_core_repo(active_graph_store):
    # 1. Ingest Repo A (representing core shared library) into the graph store
    repo_a_nodes = [
        NodeData(
            label="Module",
            id="core_lib.math",
            properties={
                "name": "math",
                "file_path": "src/core_lib/math.py",
                "repository": "core-lib",
            },
        ),
        NodeData(
            label="Class",
            id="core_lib.math.Calculator",
            properties={
                "name": "Calculator",
                "file_path": "src/core_lib/math.py",
                "repository": "core-lib",
            },
        ),
    ]
    repo_a_edges = [
        EdgeData(src_id="core_lib.math", rel="DEFINES", dst_id="core_lib.math.Calculator")
    ]

    # Ingest core-lib graph data
    result_a = ParseResult(nodes=repo_a_nodes, edges=repo_a_edges)
    active_graph_store.ingest(result_a)

    # 2. Simulate parsing Repo B (dependent app) which imports math.Calculator
    # Repo B has a stub class "Calculator" (marked as external type because it's defined in core-lib)
    repo_b_nodes = [
        NodeData(
            label="Module",
            id="main_app.entry",
            properties={
                "name": "entry",
                "file_path": "src/main_app/entry.py",
                "repository": "main-app",
            },
        ),
        # External Stub node representing the inherited class
        NodeData(
            label="Module",
            id="Calculator",
            properties={
                "name": "Calculator",
                "file_path": "",
                "type": "external",
                "repository": "main-app",
            },
        ),
        NodeData(
            label="Class",
            id="main_app.entry.SuperCalculator",
            properties={
                "name": "SuperCalculator",
                "file_path": "src/main_app/entry.py",
                "repository": "main-app",
            },
        ),
    ]

    repo_b_edges = [
        EdgeData(src_id="main_app.entry", rel="DEFINES", dst_id="main_app.entry.SuperCalculator"),
        # App class inherits from Calculator (external stub)
        EdgeData(src_id="main_app.entry.SuperCalculator", rel="INHERITS", dst_id="Calculator"),
    ]

    result_b = ParseResult(nodes=repo_b_nodes, edges=repo_b_edges)

    # 3. Execute Global Entity Resolution against FalkorDB
    # It should look up the stub "Calculator" in the database and resolve it to "core_lib.math.Calculator"
    resolved_b = resolve_global(result_b, active_graph_store)

    # 4. Verify cross-repo link was established
    # The stub Calculator node should be dropped
    assert not any(n.id == "Calculator" for n in resolved_b.nodes)

    # The edge should be successfully rewritten to point directly to Repo A
    inherits_edge = [e for e in resolved_b.edges if e.rel == "INHERITS"][0]
    assert inherits_edge.src_id == "main_app.entry.SuperCalculator"
    assert inherits_edge.dst_id == "core_lib.math.Calculator"


def test_qdrant_vector_scoping_payload_filters(active_vector_store):
    # 1. Prepare sample chunks from two separate repositories
    chunks = [
        {
            "node_id": "core_lib.math.Calculator::0",
            "label": "Class",
            "file_path": "src/core_lib/math.py",
            "text": "class Calculator: add two numbers",
            "embedding": [0.1] * 768,
            "repository": "core-lib",
        },
        {
            "node_id": "main_app.entry.SuperCalculator::0",
            "label": "Class",
            "file_path": "src/main_app/entry.py",
            "text": "class SuperCalculator inherits Calculator",
            "embedding": [0.1] * 768,
            "repository": "main-app",
        },
    ]

    # Upsert chunks into Qdrant
    active_vector_store.upsert(chunks)

    # 2. Search scoped to "core-lib" only
    results_core = active_vector_store.search(
        embedding=[0.1] * 768, top_k=10, filter_payload={"repository": "core-lib"}
    )

    # Should only return core-lib chunks
    assert len(results_core) == 1
    assert results_core[0]["repository"] == "core-lib"
    assert results_core[0]["node_id"] == "core_lib.math.Calculator::0"

    # 3. Search scoped to "main-app" only
    results_app = active_vector_store.search(
        embedding=[0.1] * 768, top_k=10, filter_payload={"repository": "main-app"}
    )

    # Should only return main-app chunks
    assert len(results_app) == 1
    assert results_app[0]["repository"] == "main-app"
    assert results_app[0]["node_id"] == "main_app.entry.SuperCalculator::0"
