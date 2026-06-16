"""
Unit tests for ingestion/pipeline.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.ingestion.pipeline import IndexingListener, run_indexing_pipeline


class MockListener(IndexingListener):
    def __init__(self) -> None:
        self.steps = []

    def on_step(self, step_name: str, message: str, progress: float | None = None) -> None:
        self.steps.append((step_name, message, progress))


class TestIndexingPipeline:
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
            embed_model="nomic-embed-text",
            llm_model="gemma2:9b",
            llm_extract=False,
            max_tokens=512,
            listener=listener,
        )

        # Assertions on pipeline results
        assert res["nodes_parsed"] == 2
        assert res["edges_parsed"] == 1
        assert res["nodes_upserted"] == 2
        assert res["edges_upserted"] == 1
        assert res["vectors_upserted"] == 2

        # Assertions on mock calls
        mock_parse_repo.assert_called_once()
        mock_graph_store.ingest.assert_called_once()
        mock_vector_store.upsert.assert_called_once()

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
                embed_model="nomic-embed-text",
                llm_model="gemma2:9b",
                llm_extract=False,
                max_tokens=512,
            )
            assert False, "Should raise ValueError for invalid directory"
        except ValueError as exc:
            assert "Repository path is not a directory" in str(exc)
