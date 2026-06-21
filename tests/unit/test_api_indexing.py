"""
Unit tests for the indexing REST API endpoints.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

import pytest

from hybrid_rag.api.main import app


@pytest.fixture(autouse=True)
def mock_db_components():
    """Mock database connections and components during test lifecycle."""
    with (
        patch("hybrid_rag.api.main.FalkorDBStore") as mock_falkor,
        patch("hybrid_rag.api.main.QdrantStore") as mock_qdrant,
        patch("hybrid_rag.api.main.OllamaEmbedder") as mock_ollama,
        patch("hybrid_rag.api.main.HybridRetriever") as mock_retriever,
    ):
        yield mock_falkor, mock_qdrant, mock_ollama, mock_retriever


class TestApiIndexing:
    @patch("hybrid_rag.api.main.process_indexing_task")
    @patch("pathlib.Path.is_dir", return_value=True)
    def test_trigger_index_success(self, mock_is_dir, mock_process_task):
        with TestClient(app) as client:
            payload = {
                "repo_path": "/mock/repo/path",
                "languages": ["python"],
                "repo_name": "test-repo",
                "llm_extract": False,
                "max_tokens": 512,
            }
            resp = client.post("/graph/index", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "task_id" in data
            assert data["status"] == "pending"
            assert data["repository"] == "test-repo"

            # Check it queued background task
            mock_process_task.assert_called_once()

            # Check the task was registered in app state
            task_id = data["task_id"]
            assert task_id in app.state.indexing_tasks
            task = app.state.indexing_tasks[task_id]
            assert task["repository"] == "test-repo"
            assert task["status"] == "pending"

    def test_trigger_index_invalid_path(self):
        with TestClient(app) as client:
            payload = {
                "repo_path": "/mock/invalid/path",
                "languages": ["python"],
            }
            # Directory check fails on host
            with patch("pathlib.Path.is_dir", return_value=False):
                resp = client.post("/graph/index", json=payload)
                assert resp.status_code == 400
                assert "does not exist or is not a directory" in resp.json()["detail"]

    @patch("pathlib.Path.is_dir", return_value=True)
    def test_list_and_get_tasks(self, mock_is_dir):
        with TestClient(app) as client:
            # Clear tasks
            app.state.indexing_tasks.clear()

            # Insert a mock task
            task_id = "test-task-123"
            app.state.indexing_tasks[task_id] = {
                "task_id": task_id,
                "repository": "mock-repo",
                "status": "completed",
                "created_at": "2026-06-05T12:00:00",
                "completed_at": "2026-06-05T12:05:00",
                "logs": ["Task started", "Task completed"],
                "error": None,
            }

            # Test list tasks
            resp_list = client.get("/graph/index/tasks")
            assert resp_list.status_code == 200
            list_data = resp_list.json()
            assert len(list_data) == 1
            assert list_data[0]["task_id"] == task_id
            assert list_data[0]["status"] == "completed"

            # Test get specific task
            resp_get = client.get(f"/graph/index/tasks/{task_id}")
            assert resp_get.status_code == 200
            get_data = resp_get.json()
            assert get_data["task_id"] == task_id
            assert len(get_data["logs"]) == 2

            # Test get non-existent task
            resp_get_404 = client.get("/graph/index/tasks/does-not-exist")
            assert resp_get_404.status_code == 404

    @patch("os.listdir", return_value=[".git", "file.py"])
    @patch("os.path.isdir", return_value=True)
    @patch("subprocess.run")
    def test_get_repository_status(self, mock_sub_run, mock_is_dir, mock_listdir):
        from unittest.mock import MagicMock
        with TestClient(app) as client:
            # Mock get_repository_metadata on the store instance
            app.state.graph_store.get_repository_metadata.return_value = {
                "last_indexed_commit": "commit123",
                "repo_path": "/mock/repo"
            }
            
            # Mock subprocess run to return HEAD commit
            mock_head_res = MagicMock()
            mock_head_res.stdout = "commit123\n"
            mock_sub_run.return_value = mock_head_res

            # Clear active tasks to ensure no active task
            app.state.indexing_tasks.clear()

            resp = client.get("/graph/repositories/my-repo/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["repository"] == "my-repo"
            assert data["last_indexed_commit"] == "commit123"
            assert data["repo_path"] == "/mock/repo"
            assert data["head_commit"] == "commit123"
            assert data["effective_path"] == "/mock/repo"
            assert data["path_status"] == "valid"
            assert data["is_sync"] is True
            assert data["active_task"] is None

            # Test mismatch (Out of sync)
            mock_head_res.stdout = "commit999\n"
            resp = client.get("/graph/repositories/my-repo/status")
            assert resp.status_code == 200
            assert resp.json()["is_sync"] is False

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("subprocess.run")
    def test_get_repository_status_autoscan(self, mock_sub_run, mock_is_dir, mock_listdir):
        from unittest.mock import MagicMock
        with TestClient(app) as client:
            # Mock get_repository_metadata returning None (meaning not set/configured)
            app.state.graph_store.get_repository_metadata.return_value = None

            # Reset mock calls on set_repository_commit
            app.state.graph_store.set_repository_commit.reset_mock()

            # Side effects to mock paths
            def isdir_side_effect(path):
                if path == "/codebases" or path == "/codebases/my-repo" or path == "/codebases/my-repo/.git":
                    return True
                return False
            mock_is_dir.side_effect = isdir_side_effect

            def listdir_side_effect(path):
                if path == "/codebases":
                    return ["my-repo"]
                if path == "/codebases/my-repo":
                    return [".git", "file.py"]
                return []
            mock_listdir.side_effect = listdir_side_effect

            # Mock subprocess run to return HEAD commit
            mock_head_res = MagicMock()
            mock_head_res.stdout = "commitabc\n"
            mock_sub_run.return_value = mock_head_res

            # Clear active tasks to ensure no active task
            app.state.indexing_tasks.clear()

            resp = client.get("/graph/repositories/my-repo/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["repository"] == "my-repo"
            assert data["last_indexed_commit"] is None
            assert data["repo_path"] == "/codebases/my-repo"
            assert data["effective_path"] == "/codebases/my-repo"
            assert data["head_commit"] == "commitabc"
            assert data["path_status"] == "valid"
            assert data["is_sync"] is False

            # Verify set_repository_commit was called automatically to save the path
            app.state.graph_store.set_repository_commit.assert_called_once_with(
                "my-repo", "", repo_path="/codebases/my-repo"
            )

    def test_abort_indexing_task(self):
        with TestClient(app) as client:
            # Clear tasks
            app.state.indexing_tasks.clear()

            # Insert a mock task in pending state
            task_id = "test-abort-task-123"
            app.state.indexing_tasks[task_id] = {
                "task_id": task_id,
                "repository": "mock-repo",
                "status": "pending",
                "created_at": "2026-06-05T12:00:00",
                "completed_at": None,
                "logs": ["Task initialized and queued."],
                "error": None,
                "progress": 0.0,
                "current_step": "init",
                "current_message": "Task queued.",
            }

            # Test abort success
            resp = client.post(f"/graph/index/tasks/{task_id}/abort")
            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] == task_id
            assert data["status"] == "aborted"
            
            # Verify status in state is updated
            assert app.state.indexing_tasks[task_id]["status"] == "aborted"
            assert app.state.indexing_tasks[task_id]["completed_at"] is not None

            # Test aborting non-existent task
            resp_404 = client.post("/graph/index/tasks/does-not-exist/abort")
            assert resp_404.status_code == 404

    @patch("httpx.AsyncClient.get")
    def test_list_llm_models_success(self, mock_get):
        from unittest.mock import MagicMock
        with TestClient(app) as client:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "models": [
                    {
                        "name": "qwen2.5-coder:7b",
                        "size": 4700000000,
                        "details": {
                            "parameter_size": "7B"
                        }
                    },
                    {
                        "name": "nomic-embed-text:latest",
                        "size": 274000000,
                        "details": {
                            "parameter_size": "274M"
                        }
                    },
                    {
                        "name": "gemma2:9b",
                        "size": 5400000000,
                        "details": {
                            "parameter_size": "9B"
                        }
                    }
                ]
            }
            mock_get.return_value = mock_resp

            resp = client.get("/llm/models")
            assert resp.status_code == 200
            data = resp.json()

            # nomic-embed-text should be filtered out
            assert len(data) == 2
            assert data[0]["name"] == "qwen2.5-coder:7b"
            assert data[0]["parameter_size"] == "7B"
            assert data[0]["size_bytes"] == 4700000000
            assert data[1]["name"] == "gemma2:9b"
            assert data[1]["parameter_size"] == "9B"
            assert data[1]["size_bytes"] == 5400000000
