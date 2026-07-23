from unittest.mock import MagicMock

import pytest

from hybrid_rag.graph.community_builder import CommunityBuilder
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.ports.embedder import BaseEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.vector.qdrant_store import QdrantStore


@pytest.fixture
def active_graph_store():
    # Attempt to connect to local active FalkorDB
    try:
        store = FalkorDBStore(host="localhost", port=6379, graph_name="test_global_graph_rag")
        store.clear()
        return store
    except Exception:
        pytest.skip("Local FalkorDB is not active on localhost:6379, skipping integration tests.")


def test_community_builder_partition_and_writeback(active_graph_store, monkeypatch):
    # 1. Ingest a mock codebase with two disconnected/tightly-clustered modules
    # Cluster 1: Database Operations
    db_nodes = [
        NodeData(
            label="Module", id="app.db", properties={"name": "db", "file_path": "app/db/db.py"}
        ),
        NodeData(
            label="Class",
            id="app.db.Database",
            properties={"name": "Database", "file_path": "app/db/db.py"},
        ),
        NodeData(
            label="Function",
            id="app.db.Database.connect",
            properties={"name": "connect", "file_path": "app/db/db.py"},
        ),
    ]
    db_edges = [
        EdgeData(src_id="app.db", rel="DEFINES", dst_id="app.db.Database"),
        EdgeData(src_id="app.db.Database", rel="DEFINES", dst_id="app.db.Database.connect"),
    ]

    # Cluster 2: UI Router
    ui_nodes = [
        NodeData(
            label="Module", id="app.ui", properties={"name": "ui", "file_path": "app/ui/ui.py"}
        ),
        NodeData(
            label="Class",
            id="app.ui.Router",
            properties={"name": "Router", "file_path": "app/ui/ui.py"},
        ),
        NodeData(
            label="Function",
            id="app.ui.Router.render",
            properties={"name": "render", "file_path": "app/ui/ui.py"},
        ),
    ]
    ui_edges = [
        EdgeData(src_id="app.ui", rel="DEFINES", dst_id="app.ui.Router"),
        EdgeData(src_id="app.ui.Router", rel="DEFINES", dst_id="app.ui.Router.render"),
    ]

    # Bridge Edge linking Cluster 1 to Cluster 2
    bridge_edge = [
        EdgeData(src_id="app.ui.Router.render", rel="CALLS", dst_id="app.db.Database.connect")
    ]

    result = ParseResult(nodes=db_nodes + ui_nodes, edges=db_edges + ui_edges + bridge_edge)

    active_graph_store.ingest(result)

    # 2. Mock Ollama LLM Report Generation to run deterministically and offline
    def mock_generate_report(self, comm_id, prompt):
        if comm_id.endswith("_0"):
            return (
                "Database Layer",
                "This community manages connections and raw read/write database operations.",
            )
        else:
            return (
                "UI Routing System",
                "This community handles routing, rendering, and web requests.",
            )

    monkeypatch.setattr(CommunityBuilder, "_generate_community_report", mock_generate_report)

    # 3. Build Communities using directory structure partitioning
    builder = CommunityBuilder(graph_store=active_graph_store)
    count = builder.build_communities()

    # Assert that it successfully partitioned the nodes into 2 main communities (by directory path)
    assert count == 2

    # 4. Verify Communities in FalkorDB
    # Fetch Community nodes
    comm_nodes_res = active_graph_store.query(
        "MATCH (c:Community) RETURN c.id, c.name, c.summary ORDER BY c.id"
    )
    comm_nodes = comm_nodes_res.result_set
    assert len(comm_nodes) == 2
    assert comm_nodes[0][0] == "community_lvl_0_0"
    assert comm_nodes[1][0] == "community_lvl_0_1"

    # Fetch IN_COMMUNITY links to verify code nodes are partitioned
    links_res = active_graph_store.query("MATCH (n)-[:IN_COMMUNITY]->(c:Community) RETURN count(n)")
    assert links_res.result_set[0][0] == 6  # All 6 code nodes linked

    # Fetch COMMUNITY_DEPENDS cross-community edge
    deps_res = active_graph_store.query(
        "MATCH (a:Community)-[r:COMMUNITY_DEPENDS]->(b:Community) RETURN r.weight"
    )
    assert len(deps_res.result_set) == 1
    assert deps_res.result_set[0][0] == 1  # 1 bridge CALLS edge

    # 5. Verify Retrieval using HybridRetriever
    mock_embedder = MagicMock(spec=BaseEmbedder)
    mock_vector_store = MagicMock(spec=QdrantStore)

    retriever = HybridRetriever(
        graph_store=active_graph_store, vector_store=mock_vector_store, embedder=mock_embedder
    )

    # Execute a global query that should trigger Global Search Routing
    global_results = retriever.retrieve("Tóm tắt cấu trúc và thiết kế hệ thống", top_k=10)

    # Check that it returns the community summaries instead of raw vector chunks
    assert len(global_results) == 2
    assert any("Database" in r["name"] for r in global_results)
    assert any("UI Routing" in r["name"] for r in global_results)
    assert all(r["label"] == "Community" for r in global_results)
