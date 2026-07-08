"""
Unit tests for ingestion/ollama_embedder.py.
All HTTP calls are mocked — no Ollama service required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.ingestion.parser import NodeData


def _mock_response(embedding: list[float]) -> MagicMock:
    """Return a mock httpx.Response yielding JSON with an embedding list."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"embedding": embedding}
    return mock_resp


def _mock_batch_response(embeddings: list[list[float]]) -> MagicMock:
    """Return a mock httpx.Response yielding JSON with multiple embeddings."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"embeddings": embeddings}
    return mock_resp


def _embedder() -> OllamaEmbedder:
    return OllamaEmbedder(ollama_url="http://localhost:11434", model="test-embed-model")


class TestOllamaEmbedder:
    def test_embed_query_success(self):
        embedder = _embedder()
        expected = [0.1] * 768
        with patch.object(
            embedder._client, "post", return_value=_mock_response(expected)
        ) as mock_post:
            res = embedder.embed_query("hello world")
            assert res == expected
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert kwargs["json"] == {
                "model": "test-embed-model",
                "prompt": "hello world",
                "keep_alive": "10s",
            }

    def test_embed_texts_batch_success(self):
        embedder = _embedder()
        expected = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        mock_resp = _mock_batch_response(expected)
        with patch.object(embedder._client, "post", return_value=mock_resp) as mock_post:
            res = embedder.embed_texts(["one", "two", "three"])
            assert res == expected
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0].endswith("/api/embed")
            assert kwargs["json"]["input"] == ["one", "two", "three"]

    def test_embed_texts_concurrent_fallback(self):
        embedder = _embedder()
        expected = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        # First call is /api/embed (fails), subsequent ones are fallback /api/embeddings (success)
        mock_posts = [
            Exception("Batch embed not supported"),
            _mock_response([0.1] * 768),
            _mock_response([0.2] * 768),
            _mock_response([0.3] * 768),
        ]

        def side_effect(*args, **kwargs):
            val = mock_posts.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch.object(embedder._client, "post", side_effect=side_effect) as mock_post:
            res = embedder.embed_texts(["one", "two", "three"])
            assert res == expected
            assert mock_post.call_count == 4

    def test_embed_texts_concurrency_one_fallback(self, monkeypatch):
        # Set EMBED_CONCURRENCY to "1" to test the sequential branch
        monkeypatch.setenv("EMBED_CONCURRENCY", "1")
        embedder = _embedder()
        expected = [[0.1] * 768, [0.2] * 768]

        # First call is /api/embed (fails), subsequent ones are fallback /api/embeddings (success)
        mock_posts = [
            Exception("Batch embed not supported"),
            _mock_response([0.1] * 768),
            _mock_response([0.2] * 768),
        ]

        def side_effect(*args, **kwargs):
            val = mock_posts.pop(0)
            if isinstance(val, Exception):
                raise val
            return val

        with patch.object(embedder._client, "post", side_effect=side_effect) as mock_post:
            res = embedder.embed_texts(["one", "two"])
            assert res == expected
            assert mock_post.call_count == 3

    def test_embed_failsafe_permanent_failure(self):
        embedder = _embedder()

        # Mock connection failure on every attempt
        with patch.object(
            embedder._client, "post", side_effect=Exception("Connection refused")
        ) as mock_post:
            with patch("time.sleep") as mock_sleep:  # Mock sleep so tests are fast
                res = embedder.embed_query("hello")

                # Should return zero-vector of length 768
                assert res == [0.0] * 768
                # 3 regular attempts + 1 fallback attempt = 4 total attempts
                assert mock_post.call_count == 4
                # Should have slept between retries (2 sleeps)
                assert mock_sleep.call_count == 2

    def test_embed_nodes_filter_external_stubs(self):
        embedder = _embedder()
        nodes = [
            NodeData(id="mod1", label="Module", properties={"file_path": "a.py", "type": "local"}),
            NodeData(
                id="mod2", label="Module", properties={"file_path": "", "type": "external"}
            ),  # External stub
            NodeData(id="func1", label="Function", properties={"name": "foo", "file_path": "a.py"}),
        ]

        expected = [0.1] * 768
        with patch.object(embedder._client, "post", return_value=_mock_response(expected)):
            chunks = embedder.embed_nodes(nodes)

            # The external stub module is skipped, so only 2 chunks remain
            assert len(chunks) == 2
            assert chunks[0]["node_id"] == "mod1"
            assert chunks[0]["embedding"] == expected
            assert chunks[1]["node_id"] == "func1"
            assert chunks[1]["embedding"] == expected

    def test_context_manager_closes_client(self):
        embedder = _embedder()
        with patch.object(embedder._client, "close") as mock_close:
            with embedder:
                pass
            mock_close.assert_called_once()
