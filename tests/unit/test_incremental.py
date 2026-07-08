"""
Unit tests for incremental indexing and store cleanup APIs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from hybrid_rag.constants import DEFAULT_LLM_MODEL
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.ingestion.pipeline import _detect_git_changes, run_indexing_pipeline
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

        # Test set_repository_commit with path
        store.set_repository_commit("repo123", "commit456", "/path/to/repo")
        mock_graph.query.assert_called_with(
            "MERGE (r:RepositoryMetadata {id: $repo}) SET r.last_indexed_commit = $commit_hash, r.repo_path = $repo_path, r.updated_at = timestamp()",
            {"repo": "repo123", "commit_hash": "commit456", "repo_path": "/path/to/repo"},
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
        mock_meta_res = MagicMock()
        mock_meta_res.result_set = [["commit456", "/path/to/repo", 1700000000000]]
        mock_graph.query.return_value = mock_meta_res

        meta = store.get_repository_metadata("repo123")
        assert meta == {
            "last_indexed_commit": "commit456",
            "repo_path": "/path/to/repo",
            "updated_at": 1700000000000,
        }
        mock_graph.query.assert_called_with(
            "MATCH (r:RepositoryMetadata {id: $repo}) RETURN r.last_indexed_commit AS commit, r.repo_path AS path, r.updated_at AS updated_at",
            {"repo": "repo123"},
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


@patch("hybrid_rag.ingestion.pipeline._detect_git_changes")
@patch("hybrid_rag.ingestion.parser.parse_file")
@patch("subprocess.run")
def test_incremental_indexing_pipeline_run(
    mock_sub_run, mock_parse_file, mock_detect_git, tmp_path
):
    # Set up mocks
    mock_graph = MagicMock()
    mock_vector = MagicMock()

    # last indexed commit in DB is commit123, HEAD is commit456
    mock_graph.get_repository_commit.return_value = "commit123"

    mock_head_res = MagicMock()
    mock_head_res.stdout = "commit456\n"
    mock_sub_run.return_value = mock_head_res

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

        # Assert cleanup was called
        # both src/b.py and src/a.py should be cleaned up
        mock_graph.delete_file_nodes.assert_any_call("src/a.py", "myrepo")
        mock_graph.delete_file_nodes.assert_any_call("src/b.py", "myrepo")
        mock_vector.delete_file_vectors.assert_any_call("src/a.py", "myrepo")
        mock_vector.delete_file_vectors.assert_any_call("src/b.py", "myrepo")

        # Assert only src/a.py was parsed
        mock_parse_file.assert_called_once_with(tmp_path / "src/a.py", tmp_path, repo_name="myrepo")

        # Assert commit was updated to HEAD (commit456)
        mock_graph.set_repository_commit.assert_called_with(
            "myrepo", "commit456", repo_path=str(tmp_path)
        )
