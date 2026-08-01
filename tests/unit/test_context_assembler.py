import pytest

from hybrid_rag.retrieval.context_assembler import ContextAssembler


@pytest.fixture
def sample_results():
    return [
        {
            "node_id": "module.math_utils",
            "name": "math_utils",
            "label": "Module",
            "file_path": "src/math_utils.py",
            "text": "def add(a, b): return a + b",
            "source": "vector",
            "rrf_score": 0.05,
        },
        {
            "node_id": "module.math_utils.Calculator",
            "name": "Calculator",
            "label": "Class",
            "file_path": "src/math_utils.py",
            "text": "class Calculator:\n    def mul(self, a, b):\n        return a * b",
            "source": "graph",
            "rrf_score": 0.04,
        },
        {
            "node_id": "module.math_utils.Calculator.div",
            "name": "div",
            "label": "Function",
            "file_path": "src/math_utils.py",
            "text": "def div(self, a, b):\n    return a / b",
            "source": "hybrid",
            "rrf_score": 0.03,
        },
    ]


def test_context_assembler_fallback_top_n(sample_results):
    assembler = ContextAssembler()
    # Should use legacy fallback (top_n = 2)
    ctx = assembler.assemble(sample_results, top_n=2, query="test query")

    assert ctx.metadata["use_budget"] is False
    assert ctx.metadata["shown"] == 2
    assert len(ctx.chunks) == 2
    assert ctx.chunks[0]["name"] == "math_utils"
    assert ctx.chunks[1]["name"] == "Calculator"
    assert "Query: test query" in ctx.text
    assert "Calculator" in ctx.text
    assert "div" not in ctx.text


def test_context_assembler_max_tokens_budget(sample_results):
    assembler = ContextAssembler()

    # Each chunk formatted length:
    # Chunk 1: Header ~70 chars + text 27 chars + separator = ~100 chars -> ~25 tokens (at //4)
    # Chunk 2: Header ~80 chars + text 55 chars + separator = ~140 chars -> ~35 tokens (at //4)
    # Total tokens expected for Query Overhead + Chunk 1 = ~30 tokens.
    # If we set max_tokens=40, it should fit Query + Chunk 1, but Chunk 2 would push it over (~65 tokens total).
    ctx = assembler.assemble(sample_results, max_tokens=40, query="short query")

    assert ctx.metadata["use_budget"] is True
    assert ctx.metadata["budget_limit"] == 40
    assert ctx.metadata["shown"] == 1
    assert len(ctx.chunks) == 1
    assert ctx.chunks[0]["name"] == "math_utils"
    assert ctx.metadata["chunks_included_count"] == 1
    assert ctx.metadata["chunks_excluded_count"] == 2


def test_context_assembler_max_chars_budget(sample_results):
    assembler = ContextAssembler()

    # Query + Chunk 1 formatted character count is around 130 characters.
    # Let's set max_chars=180. It should fit Query + Chunk 1, but exclude Chunk 2 (which is ~140 characters on its own).
    ctx = assembler.assemble(sample_results, max_chars=180, query="short query")

    assert ctx.metadata["use_budget"] is True
    assert ctx.metadata["budget_limit"] == 180
    assert ctx.metadata["shown"] == 1
    assert len(ctx.chunks) == 1
    assert ctx.metadata["total_chars"] <= 180
    assert ctx.metadata["chunks_included_count"] == 1
    assert ctx.metadata["chunks_excluded_count"] == 2


def test_context_assembler_custom_token_estimator(sample_results):
    assembler = ContextAssembler()

    # Use a custom estimator that counts words in the text instead of char // 4
    def word_count_estimator(text):
        return len(text.split())

    # Let's set a token budget using this custom estimator
    ctx = assembler.assemble(
        sample_results, max_tokens=25, token_estimator=word_count_estimator, query="custom query"
    )

    assert ctx.metadata["use_budget"] is True
    # The custom estimator should be recorded and applied correctly
    assert ctx.metadata["total_tokens"] <= 25
    assert ctx.metadata["chunks_included_count"] >= 1


def test_context_assembler_keeps_same_path_from_different_repositories(monkeypatch):
    from hybrid_rag.config import app_config

    monkeypatch.setattr(app_config, "get_repo_path", lambda _repo: None)
    results = [
        {
            "node_id": "repo-a::pkg.service.run::0",
            "name": "run",
            "label": "Function",
            "file_path": "src/service.py",
            "repository": "repo-a",
            "text": "return 'repo-a'",
            "source": "vector",
            "rrf_score": 0.05,
        },
        {
            "node_id": "repo-b::pkg.service.run::0",
            "name": "run",
            "label": "Function",
            "file_path": "src/service.py",
            "repository": "repo-b",
            "text": "return 'repo-b'",
            "source": "vector",
            "rrf_score": 0.04,
        },
    ]

    ctx = ContextAssembler().assemble(results, top_n=5)

    assert [chunk["repository"] for chunk in ctx.chunks] == ["repo-a", "repo-b"]
    assert "return 'repo-a'" in ctx.text
    assert "return 'repo-b'" in ctx.text


def test_context_assembler_skeletonized(tmp_path, monkeypatch):
    from hybrid_rag.config import app_config

    # Write a dummy python file
    code = """
class TargetClass:
    def target_method(self):
        return 42

    def sibling_method(self):
        return 24
"""
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    file_path = "src/math_utils.py"
    abs_file_path = repo_dir / file_path
    abs_file_path.parent.mkdir(parents=True, exist_ok=True)
    abs_file_path.write_text(code, encoding="utf-8")

    # Mock config repo path
    monkeypatch.setattr(
        app_config, "get_repo_path", lambda repo: str(repo_dir) if repo == "my_repo" else None
    )

    assembler = ContextAssembler()
    results = [
        {
            "node_id": "module.math_utils.TargetClass.target_method",
            "name": "target_method",
            "label": "Function",
            "file_path": file_path,
            "text": "def target_method(self):\n    return 42",
            "source": "vector",
            "repository": "my_repo",
            "rrf_score": 0.05,
        }
    ]

    ctx = assembler.assemble(results, top_n=5, query="test query")

    # Sibling method should be skeletonized/collapsed
    assert "[Skeletonized]" in ctx.text
    assert "def sibling_method(self):" in ctx.text
    assert "..." in ctx.text
    assert "return 24" not in ctx.text

    # Target method should be fully intact
    assert "def target_method(self):" in ctx.text
    assert "return 42" in ctx.text
