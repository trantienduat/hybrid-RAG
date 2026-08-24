"""
Unit tests for ingestion/pipeline.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.constants import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL
from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.ingestion.pipeline import (
    IndexingListener,
    _embedding_batch_size,
    run_indexing_pipeline,
)


class MockListener(IndexingListener):
    def __init__(self) -> None:
        self.steps = []

    def on_step(self, step_name: str, message: str, progress: float | None = None) -> None:
        self.steps.append((step_name, message, progress))


class TestIndexingPipeline:
    def test_embedding_batch_size_reads_positive_environment_value(self, monkeypatch):
        monkeypatch.setenv("EMBED_BATCH_SIZE", "16")

        assert _embedding_batch_size() == 16

    @patch("hybrid_rag.ingestion.parser.parse_repo")
    @patch("hybrid_rag.ingestion.entity_resolver.resolve")
    @patch("hybrid_rag.ingestion.entity_resolver.resolve_global")
    @patch("hybrid_rag.ingestion.ollama_embedder.OllamaEmbedder")
    @patch("hybrid_rag.ingestion.chunker.chunk_file")
    def test_run_indexing_pipeline_success(
        self,
        mock_chunk_file,
        mock_embedder_class,
        mock_resolve_global,
        mock_resolve,
        mock_parse_repo,
        tmp_path,
    ):
        # 1. Mock AST parsing result
        mock_nodes = [
            NodeData(id="mod1", label="Module", properties={"file_path": "a.py"}),
            NodeData(id="func1", label="Function", properties={"name": "foo", "file_path": "a.py"}),
        ]
        mock_edges = [EdgeData(src_id="mod1", rel="CONTAINS", dst_id="func1")]
        mock_parse_repo.return_value = ParseResult(nodes=mock_nodes, edges=mock_edges, errors=[])

        # 2. Mock entity resolution
        mock_resolve.return_value = mock_parse_repo.return_value
        mock_resolve_global.return_value = mock_parse_repo.return_value

        # 3. Mock Stores
        mock_graph_store = MagicMock()
        mock_graph_store.ingest.return_value = {"nodes": 2, "edges": 1}

        mock_vector_store = MagicMock()
        mock_vector_store.upsert.return_value = 2

        # 4. Mock chunking
        mock_chunk = MagicMock()
        mock_chunk.node_id = "func1"
        mock_chunk.chunk_index = 0
        mock_chunk.text = "def foo():\n    pass"
        mock_chunk.label = "Function"
        mock_chunk.file_path = "a.py"
        mock_chunk_file.return_value = [mock_chunk]

        # 5. Mock OllamaEmbedder context manager & embedding
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 768]
        mock_embedder_class.return_value.__enter__.return_value = mock_embedder

        # Create temporary directory
        tmp_dir = tmp_path / "repo"
        tmp_dir.mkdir()
        (tmp_dir / "a.py").touch()

        # Run pipeline
        listener = MockListener()
        res = run_indexing_pipeline(
            repo_path=tmp_dir,
            languages=["python"],
            repo_name="test-repo",
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            ollama_url="http://localhost:11434",
            embed_model=DEFAULT_EMBED_MODEL,
            llm_model=DEFAULT_LLM_MODEL,
            llm_extract=False,
            max_tokens=512,
            listener=listener,
        )

        # Assertions on pipeline results
        assert res["nodes_parsed"] == 3
        assert res["edges_parsed"] == 1
        assert res["nodes_upserted"] == 2
        assert res["edges_upserted"] == 1
        assert res["vectors_upserted"] == 2

        # Assertions on mock calls
        mock_parse_repo.assert_called_once()
        mock_graph_store.ingest.assert_called_once()
        mock_vector_store.upsert.assert_called_once()
        mock_chunk_file.assert_called_once_with(
            tmp_dir / "a.py",
            tmp_dir,
            max_tokens=512,
            repo_name="test-repo",
        )
        payload = mock_vector_store.upsert.call_args.args[0][0]
        assert payload["name"] == "foo"
        assert payload["indexed_commit"] == ""
        assert payload["source_identity"].startswith("sha256:")
        assert payload["index_run_id"]
        metadata = mock_graph_store.set_repository_metadata.call_args.args[1]
        assert metadata["source_kind"] == "directory"
        mock_vector_store.set_repository_metadata.assert_called_once_with(
            "test-repo",
            metadata,
        )

        # Assertions on listener calls
        steps = [step[0] for step in listener.steps]
        assert "parse" in steps
        assert "resolution" in steps
        assert "db_write" in steps
        assert "embed_chunks" in steps
        assert "vector_write" in steps
        assert "complete" in steps

    def test_run_indexing_pipeline_invalid_dir(self):
        mock_graph_store = MagicMock()
        mock_vector_store = MagicMock()

        # Test directory checking
        invalid_path = Path("/does/not/exist/path/123")
        try:
            run_indexing_pipeline(
                repo_path=invalid_path,
                languages=["python"],
                repo_name="invalid",
                graph_store=mock_graph_store,
                vector_store=mock_vector_store,
                ollama_url="http://localhost:11434",
                embed_model=DEFAULT_EMBED_MODEL,
                llm_model=DEFAULT_LLM_MODEL,
                llm_extract=False,
                max_tokens=512,
            )
            assert False, "Should raise ValueError for invalid directory"
        except ValueError as exc:
            assert "Repository path is not a directory" in str(exc)

    def test_pipeline_rejects_java_before_store_access(self, tmp_path):
        (tmp_path / "Main.java").write_text("class Main {}")
        graph_store = MagicMock()
        vector_store = MagicMock()

        with pytest.raises(ValueError, match="Only Python indexing"):
            run_indexing_pipeline(
                repo_path=tmp_path,
                languages=["java"],
                repo_name="java-repo",
                graph_store=graph_store,
                vector_store=vector_store,
                ollama_url="http://localhost:11434",
                embed_model=DEFAULT_EMBED_MODEL,
                llm_model=DEFAULT_LLM_MODEL,
                llm_extract=False,
                max_tokens=512,
            )

        graph_store.get_repository_metadata.assert_not_called()
        graph_store.ingest.assert_not_called()
        vector_store.upsert.assert_not_called()

    def test_pipeline_rejects_java_only_repo_with_default_language(self, tmp_path):
        (tmp_path / "Main.java").write_text("class Main {}")

        with pytest.raises(ValueError, match="Java indexing is not implemented"):
            run_indexing_pipeline(
                repo_path=tmp_path,
                languages=["python"],
                repo_name="java-repo",
                graph_store=MagicMock(),
                vector_store=MagicMock(),
                ollama_url="http://localhost:11434",
                embed_model=DEFAULT_EMBED_MODEL,
                llm_model=DEFAULT_LLM_MODEL,
                llm_extract=False,
                max_tokens=512,
            )

    @patch("hybrid_rag.ingestion.parser.parse_repo", side_effect=RuntimeError("parse failed"))
    def test_rebuild_preserves_existing_index_when_parse_fails(self, _mock_parse, tmp_path):
        graph_store = MagicMock()
        vector_store = MagicMock()

        try:
            run_indexing_pipeline(
                repo_path=tmp_path,
                languages=["python"],
                repo_name="repo-one",
                graph_store=graph_store,
                vector_store=vector_store,
                ollama_url="http://localhost:11434",
                embed_model=DEFAULT_EMBED_MODEL,
                llm_model=DEFAULT_LLM_MODEL,
                llm_extract=False,
                max_tokens=512,
                rebuild=True,
            )
            assert False, "Expected parsing failure"
        except RuntimeError as exc:
            assert str(exc) == "parse failed"

        graph_store.delete_repository.assert_not_called()
        vector_store.delete_repository.assert_not_called()
        graph_store.delete_repository_except_run.assert_not_called()
        vector_store.delete_repository_except_run.assert_not_called()

    @patch("hybrid_rag.ingestion.parser.parse_repo")
    @patch("hybrid_rag.ingestion.entity_resolver.resolve", side_effect=lambda result: result)
    @patch(
        "hybrid_rag.ingestion.entity_resolver.resolve_global",
        side_effect=lambda result, _store: result,
    )
    @patch("hybrid_rag.ingestion.chunker.chunk_file")
    @patch("hybrid_rag.ingestion.ollama_embedder.OllamaEmbedder")
    def test_rebuild_preserves_existing_index_when_embedding_fails(
        self,
        mock_embedder_class,
        mock_chunk_file,
        _mock_resolve_global,
        _mock_resolve,
        mock_parse_repo,
        tmp_path,
    ):
        mock_parse_repo.return_value = ParseResult(
            nodes=[NodeData(label="Function", id="repo::mod.run", properties={"file_path": "a.py"})]
        )
        chunk = MagicMock(
            node_id="repo::mod.run",
            chunk_index=0,
            text="def run(): pass",
            label="Function",
            file_path="a.py",
        )
        mock_chunk_file.return_value = [chunk]
        mock_embedder_class.return_value.__enter__.return_value.embed_texts.side_effect = (
            RuntimeError("embedding failed")
        )
        graph_store = MagicMock()
        vector_store = MagicMock()

        with pytest.raises(RuntimeError, match="embedding failed"):
            run_indexing_pipeline(
                repo_path=tmp_path,
                languages=["python"],
                repo_name="repo-one",
                graph_store=graph_store,
                vector_store=vector_store,
                ollama_url="http://localhost:11434",
                embed_model=DEFAULT_EMBED_MODEL,
                llm_model=DEFAULT_LLM_MODEL,
                llm_extract=False,
                max_tokens=512,
                rebuild=True,
            )

        graph_store.ingest.assert_not_called()
        vector_store.upsert.assert_not_called()
        graph_store.delete_repository_except_run.assert_not_called()
        vector_store.delete_repository_except_run.assert_not_called()

    @patch("hybrid_rag.ingestion.parser.parse_repo")
    @patch("hybrid_rag.ingestion.entity_resolver.resolve", side_effect=lambda result: result)
    @patch(
        "hybrid_rag.ingestion.entity_resolver.resolve_global",
        side_effect=lambda result, _store: result,
    )
    def test_rebuild_deletes_only_repository_namespace(
        self,
        _mock_resolve_global,
        _mock_resolve,
        mock_parse_repo,
        tmp_path,
    ):
        mock_parse_repo.return_value = ParseResult()
        graph_store = MagicMock()
        graph_store.ingest.return_value = {"nodes": 0, "edges": 0}
        vector_store = MagicMock()

        run_indexing_pipeline(
            repo_path=tmp_path,
            languages=["python"],
            repo_name="repo-one",
            graph_store=graph_store,
            vector_store=vector_store,
            ollama_url="http://localhost:11434",
            embed_model=DEFAULT_EMBED_MODEL,
            llm_model=DEFAULT_LLM_MODEL,
            llm_extract=False,
            max_tokens=512,
            rebuild=True,
        )

        metadata = graph_store.set_repository_metadata.call_args.args[1]
        graph_store.delete_repository_except_run.assert_called_once_with(
            "repo-one", metadata["index_run_id"]
        )
        vector_store.delete_repository_except_run.assert_called_once_with(
            "repo-one", metadata["index_run_id"]
        )
