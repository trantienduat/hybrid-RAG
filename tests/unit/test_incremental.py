"""
Unit tests for incremental indexing and store cleanup APIs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from qdrant_client.models import PayloadSchemaType

from hybrid_rag.constants import DEFAULT_LLM_MODEL
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.ingestion.pipeline import (
    INDEX_SCHEMA_VERSION,
    _detect_git_changes,
    run_indexing_pipeline,
)
from hybrid_rag.vector.qdrant_store import QdrantStore


def test_detect_git_changes_success():
    repo = Path("/mock/repo")
    last_commit = "commit123"

    mock_diff = "M\tsrc/foo.py\nD\tsrc/bar.py\nA\tsrc/baz.py\nR099\tsrc/old.py\tsrc/new.py\n"
    mock_status = "?? src/untracked.py\n?? docs/untracked.txt\n"

    def mock_subprocess_run(args, cwd, **kwargs):
        cmd = args[1]
        mock_res = MagicMock()
        mock_res.stdout = ""
        if cmd == "rev-parse" and args[2] == "--is-inside-work-tree":
            mock_res.stdout = "true\n"
        elif cmd == "cat-file":
            mock_res.stdout = "commit\n"
        elif cmd == "diff":
            mock_res.stdout = mock_diff
        elif cmd == "status":
            mock_res.stdout = mock_status
        return mock_res

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        res = _detect_git_changes(
            repo=repo,
            last_commit=last_commit,
            languages=["python"],
            excludes=None,
        )
        assert res is not None
        modified, deleted = res

        # bar.py and old.py should be deleted
        assert deleted == {"src/bar.py", "src/old.py"}
        # foo.py, baz.py, new.py, untracked.py should be modified
        # docs/untracked.txt should be filtered out by extension
        assert modified == {"src/foo.py", "src/baz.py", "src/new.py", "src/untracked.py"}


def test_detect_git_changes_not_git_repo():
    repo = Path("/mock/repo")

    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
        res = _detect_git_changes(
            repo=repo,
            last_commit="commit123",
            languages=["python"],
            excludes=None,
        )
        assert res is None


def test_falkordb_store_commit_metadata():
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    with patch("falkordb.FalkorDB", return_value=mock_db):
        store = FalkorDBStore(host="localhost", port=6379, graph_name="test")

        # Test set_repository_commit without path
        store.set_repository_commit("repo123", "commit456")
        mock_graph.query.assert_called_with(
            "MERGE (r:RepositoryMetadata {id: $repo}) SET r.last_indexed_commit = $commit_hash, r.updated_at = timestamp()",
            {"repo": "repo123", "commit_hash": "commit456"},
        )

        # Test get_repository_commit
        mock_result = MagicMock()
        mock_result.result_set = [["commit456"]]
        mock_graph.query.return_value = mock_result

        commit = store.get_repository_commit("repo123")
        assert commit == "commit456"
        mock_graph.query.assert_called_with(
            "MATCH (r:RepositoryMetadata {id: $repo}) RETURN r.last_indexed_commit AS commit",
            {"repo": "repo123"},
        )

        # Test get_repository_metadata
        mock_node = MagicMock()
        mock_node.properties = {
            "last_indexed_commit": "commit456",
            "source_identity": "git:repo@commit456",
            "updated_at": 1700000000000,
        }
        mock_meta_res = MagicMock()
        mock_meta_res.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_meta_res

        meta = store.get_repository_metadata("repo123")
        assert meta == {
            "last_indexed_commit": "commit456",
            "source_identity": "git:repo@commit456",
            "updated_at": 1700000000000,
        }
        mock_graph.query.assert_called_with(
            "MATCH (r:RepositoryMetadata {id: $repo}) RETURN r",
            {"repo": "repo123"},
        )

        store.set_repository_metadata(
            "repo123",
            {"last_indexed_commit": "commit456", "index_run_id": "run-1"},
        )
        mock_graph.query.assert_called_with(
            "MERGE (r:RepositoryMetadata {id: $repo}) "
            "SET r += $metadata, r.updated_at = timestamp()",
            {
                "repo": "repo123",
                "metadata": {
                    "last_indexed_commit": "commit456",
                    "index_run_id": "run-1",
                },
            },
        )


def test_falkordb_store_delete_file_nodes():
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    with patch("falkordb.FalkorDB", return_value=mock_db):
        store = FalkorDBStore(host="localhost", port=6379, graph_name="test")
        store.delete_file_nodes("src/foo.py", "repo123")
        mock_graph.query.assert_called_with(
            "MATCH (n) WHERE n.file_path = $file_path AND n.repository = $repository DETACH DELETE n",
            {"file_path": "src/foo.py", "repository": "repo123"},
        )

        store.delete_repository("repo123")
        mock_graph.query.assert_called_with(
            "MATCH (n) WHERE n.repository = $repository OR "
            "(n:RepositoryMetadata AND n.id = $repository) DETACH DELETE n",
            {"repository": "repo123"},
        )


def test_falkordb_store_deletes_only_stale_run_records():
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    with patch("falkordb.FalkorDB", return_value=mock_db):
        store = FalkorDBStore(host="localhost", port=6379, graph_name="test")
        store.delete_repository_except_run("repo123", "run-new")

    assert mock_graph.query.call_count == 2
    for call in mock_graph.query.call_args_list:
        assert "coalesce" in call.args[0]
        assert call.args[1] == {"repository": "repo123", "index_run_id": "run-new"}


def test_qdrant_store_delete_file_vectors():
    mock_client = MagicMock()
    with patch("hybrid_rag.vector.qdrant_store.QdrantClient", return_value=mock_client):
        store = QdrantStore(host="localhost", port=6333, collection="test_col")
        store.delete_file_vectors("src/foo.py", "repo123")

        # Check client delete call
        mock_client.delete.assert_called_once()
        args, kwargs = mock_client.delete.call_args
        assert kwargs["collection_name"] == "test_col"
        filter_obj = kwargs["points_selector"]

        # Verify it has must conditions matching file_path and repository
        conditions = filter_obj.must
        assert len(conditions) == 2
        assert conditions[0].key == "file_path"
        assert conditions[0].match.value == "src/foo.py"
        assert conditions[1].key == "repository"
        assert conditions[1].match.value == "repo123"


def test_qdrant_store_repository_metadata():
    mock_client = MagicMock()
    mock_client.facet.side_effect = [
        MagicMock(hits=[MagicMock(value="abc")]),
        MagicMock(hits=[MagicMock(value="run-1")]),
        MagicMock(hits=[MagicMock(value="git:repo@abc")]),
        MagicMock(hits=[MagicMock(value=INDEX_SCHEMA_VERSION)]),
        MagicMock(hits=[MagicMock(value="nomic-embed-text")]),
    ]

    with patch("hybrid_rag.vector.qdrant_store.QdrantClient", return_value=mock_client):
        store = QdrantStore(host="localhost", port=6333, collection="test_col")
        store.set_repository_metadata(
            "repo123",
            {"indexed_commit": "abc", "index_run_id": "run-1"},
        )
        metadata = store.get_repository_metadata("repo123")

    assert metadata == {
        "indexed_commit": {"abc"},
        "index_run_id": {"run-1"},
        "source_identity": {"git:repo@abc"},
        "index_schema_version": {INDEX_SCHEMA_VERSION},
        "embedding_model": {"nomic-embed-text"},
    }
    set_payload = mock_client.set_payload.call_args.kwargs
    assert set_payload["payload"]["index_run_id"] == "run-1"
    assert mock_client.facet.call_count == 5
    payload_indexes = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in mock_client.create_payload_index.call_args_list
    }
    assert payload_indexes["embedding_model"] == PayloadSchemaType.KEYWORD
    assert payload_indexes["index_schema_version"] == PayloadSchemaType.INTEGER


def test_qdrant_store_delete_repository():
    mock_client = MagicMock()
    with patch("hybrid_rag.vector.qdrant_store.QdrantClient", return_value=mock_client):
        store = QdrantStore(host="localhost", port=6333, collection="test_col")
        store.delete_repository("repo123")

    delete_call = mock_client.delete.call_args.kwargs
    assert delete_call["collection_name"] == "test_col"
    assert delete_call["points_selector"].must[0].match.value == "repo123"


def test_qdrant_store_deletes_only_stale_run_vectors():
    mock_client = MagicMock()
    with patch("hybrid_rag.vector.qdrant_store.QdrantClient", return_value=mock_client):
        store = QdrantStore(host="localhost", port=6333, collection="test_col")
        store.delete_repository_except_run("repo123", "run-new")

    selector = mock_client.delete.call_args.kwargs["points_selector"]
    assert selector.must[0].match.value == "repo123"
    assert selector.must_not[0].key == "index_run_id"
    assert selector.must_not[0].match.value == "run-new"


@patch("hybrid_rag.ingestion.pipeline._detect_git_changes")
@patch("hybrid_rag.ingestion.parser.parse_file")
@patch("hybrid_rag.ingestion.pipeline._git_source_provenance")
def test_incremental_indexing_pipeline_run(
    mock_git_provenance, mock_parse_file, mock_detect_git, tmp_path
):
    # Set up mocks
    mock_graph = MagicMock()
    mock_vector = MagicMock()

    # last indexed commit in DB is commit123, HEAD is commit456
    mock_graph.get_repository_commit.return_value = "commit123"

    mock_git_provenance.return_value = {
        "source_kind": "git",
        "source_path": str(tmp_path),
        "source_commit": "commit456",
        "source_digest": "",
        "source_identity": "git:repo@commit456",
        "working_tree_dirty": False,
    }

    # Git changes: modified: src/a.py, deleted: src/b.py
    mock_detect_git.return_value = ({"src/a.py"}, {"src/b.py"})

    # Parse result of modified file
    mock_parse_file.return_value = ParseResult(
        nodes=[
            NodeData(
                label="Function",
                id="src/a.py::foo",
                properties={"name": "foo", "file_path": "src/a.py"},
            )
        ],
        edges=[EdgeData(src_id="src/a.py", rel="DEFINES", dst_id="src/a.py::foo")],
    )

    # run pipeline
    with (
        patch("hybrid_rag.ingestion.entity_resolver.resolve", side_effect=lambda x: x),
        patch("hybrid_rag.ingestion.entity_resolver.resolve_global", side_effect=lambda x, y: x),
        patch("hybrid_rag.ingestion.chunker.chunk_file", return_value=[]),
    ):
        run_indexing_pipeline(
            repo_path=tmp_path,
            languages=["python"],
            repo_name="myrepo",
            graph_store=mock_graph,
            vector_store=mock_vector,
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            llm_model=DEFAULT_LLM_MODEL,
            llm_extract=False,
            max_tokens=512,
            incremental=True,
        )

        # Assert only src/a.py was parsed
        mock_parse_file.assert_called_once_with(tmp_path / "src/a.py", tmp_path, repo_name="myrepo")

        mock_graph.set_repository_metadata.assert_called_once()
        metadata = mock_graph.set_repository_metadata.call_args.args[1]
        assert metadata["last_indexed_commit"] == "commit456"
        assert metadata["source_identity"].endswith("@commit456")
        mock_vector.set_repository_metadata.assert_called_once_with("myrepo", metadata)

        # New records are written first; cleanup removes only stale records.
        run_id = metadata["index_run_id"]
        mock_graph.delete_file_nodes_except_run.assert_any_call("src/a.py", "myrepo", run_id)
        mock_graph.delete_file_nodes_except_run.assert_any_call("src/b.py", "myrepo", run_id)
        mock_vector.delete_file_vectors_except_run.assert_any_call("src/a.py", "myrepo", run_id)
        mock_vector.delete_file_vectors_except_run.assert_any_call("src/b.py", "myrepo", run_id)


@patch("hybrid_rag.ingestion.pipeline._detect_git_changes")
@patch("hybrid_rag.ingestion.parser.parse_repo")
@patch("hybrid_rag.ingestion.pipeline._git_source_provenance")
def test_incremental_forces_full_replacement_for_embedding_model_change(
    mock_git_provenance, mock_parse_repo, mock_detect_git, tmp_path
):
    (tmp_path / "a.py").write_text("def a(): pass")
    graph_store = MagicMock()
    graph_store.get_repository_metadata.return_value = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": "old-model",
    }
    graph_store.ingest.return_value = {"nodes": 1, "edges": 0}
    vector_store = MagicMock()
    mock_git_provenance.return_value = {
        "source_kind": "git",
        "source_path": str(tmp_path),
        "source_commit": "commit456",
        "source_digest": "",
        "source_identity": "git:repo@commit456",
        "working_tree_dirty": False,
    }
    mock_parse_repo.return_value = ParseResult()

    with (
        patch("hybrid_rag.ingestion.entity_resolver.resolve", side_effect=lambda result: result),
        patch(
            "hybrid_rag.ingestion.entity_resolver.resolve_global",
            side_effect=lambda result, _store: result,
        ),
        patch("hybrid_rag.ingestion.chunker.chunk_file", return_value=[]),
    ):
        run_indexing_pipeline(
            repo_path=tmp_path,
            languages=["python"],
            repo_name="myrepo",
            graph_store=graph_store,
            vector_store=vector_store,
            ollama_url="http://localhost:11434",
            embed_model="new-model",
            llm_model=DEFAULT_LLM_MODEL,
            llm_extract=False,
            max_tokens=512,
            incremental=True,
        )

    mock_detect_git.assert_not_called()
    mock_parse_repo.assert_called_once()
    metadata = graph_store.set_repository_metadata.call_args.args[1]
    graph_store.delete_repository_except_run.assert_called_once_with(
        "myrepo", metadata["index_run_id"]
    )
    vector_store.delete_repository_except_run.assert_called_once_with(
        "myrepo", metadata["index_run_id"]
    )
