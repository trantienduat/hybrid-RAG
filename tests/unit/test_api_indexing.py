"""
Unit tests for the indexing REST API endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Patch dependencies during app import so the lifespan doesn't start real DB clients
with (
    patch("hybrid_rag.api.main.FalkorDBStore"),
    patch("hybrid_rag.api.main.QdrantStore"),
    patch("hybrid_rag.api.main.OllamaEmbedder"),
    patch("hybrid_rag.api.main.HybridRetriever"),
):
    from hybrid_rag.api.main import app


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
