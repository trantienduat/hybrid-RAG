"""
Unit tests for GeminiEmbedder and GeminiLLMExtractor.
All HTTP calls are mocked — no Gemini API key or internet required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from hybrid_rag.ingestion.gemini_embedder import GeminiEmbedder
from hybrid_rag.ingestion.gemini_llm_extractor import GeminiLLMExtractor
from hybrid_rag.ingestion.parser import NodeData, ParseResult


def _mock_response(json_data: dict) -> MagicMock:
    """Return a mock httpx.Response yielding JSON data."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = json_data
    return mock_resp


class TestGeminiEmbedder:
    def test_embed_query_success(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        embedder = GeminiEmbedder(model="test-embed-model")

        expected_values = [0.1] * 768
        mock_data = {"embedding": {"values": expected_values}}

        with patch.object(
            embedder._client, "post", return_value=_mock_response(mock_data)
        ) as mock_post:
            res = embedder.embed_query("hello world")
            assert res == expected_values
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert "test-embed-model" in args[0]
            assert "key=test-key" in args[0]

    def test_embed_batch_success(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        embedder = GeminiEmbedder(model="test-embed-model")

        expected_values = [[0.1] * 768, [0.2] * 768]
        mock_data = {"embeddings": [{"values": [0.1] * 768}, {"values": [0.2] * 768}]}

        with patch.object(
            embedder._client, "post", return_value=_mock_response(mock_data)
        ) as mock_post:
            res = embedder.embed_texts(["one", "two"])
            assert res == expected_values
            mock_post.assert_called_once()

    def test_initialize_no_api_key_raises_error(self):
        # Explicitly pass api_key as None and ensure env is cleared
        import pytest
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
                GeminiEmbedder(api_key=None)


class TestGeminiLLMExtractor:
    def test_extractor_success(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        extractor = GeminiLLMExtractor(model="test-gemini-model")

        mock_gemini_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "edges": [
                                            {
                                                "src_id": "func1",
                                                "rel": "USES",
                                                "dst_name": "MyClass",
                                                "confidence": 0.9,
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 20,
                "totalTokenCount": 120,
            },
        }

        # Mock parsed results
        nodes = [
            NodeData(
                id="func1", label="Function", properties={"name": "my_func", "file_path": "foo.py"}
            ),
            NodeData(
                id="MyClass", label="Class", properties={"name": "MyClass", "file_path": "foo.py"}
            ),
        ]
        parse_result = ParseResult(nodes=nodes, edges=[], errors=[])

        with patch.object(
            extractor._client, "post", return_value=_mock_response(mock_gemini_json)
        ) as mock_post:
            edges = extractor.extract("def my_func(x: MyClass): pass", parse_result)
            assert len(edges) == 1
            assert edges[0].src_id == "func1"
            assert edges[0].rel == "USES"
            assert edges[0].dst_id == "MyClass"
            mock_post.assert_called_once()
