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
            "rrf_score": 0.05
        },
        {
            "node_id": "module.math_utils.Calculator",
            "name": "Calculator",
            "label": "Class",
            "file_path": "src/math_utils.py",
            "text": "class Calculator:\n    def mul(self, a, b):\n        return a * b",
            "source": "graph",
            "rrf_score": 0.04
        },
        {
            "node_id": "module.math_utils.Calculator.div",
            "name": "div",
            "label": "Function",
            "file_path": "src/math_utils.py",
            "text": "def div(self, a, b):\n    return a / b",
            "source": "hybrid",
            "rrf_score": 0.03
        }
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
    word_count_estimator = lambda text: len(text.split())
    
    # Let's set a token budget using this custom estimator
    ctx = assembler.assemble(
        sample_results, 
        max_tokens=25, 
        token_estimator=word_count_estimator, 
        query="custom query"
    )
    
    assert ctx.metadata["use_budget"] is True
    # The custom estimator should be recorded and applied correctly
    assert ctx.metadata["total_tokens"] <= 25
    assert ctx.metadata["chunks_included_count"] >= 1
