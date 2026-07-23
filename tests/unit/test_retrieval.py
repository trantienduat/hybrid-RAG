"""
Unit tests for the M3 hybrid retrieval pipeline.

No external services required — all stores and embedders are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hybrid_rag.retrieval.context_assembler import ContextAssembler, RetrievalContext
from hybrid_rag.retrieval.graph_retriever import GraphRetriever
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.retrieval.query_analyzer import QueryAnalysis, analyze
from hybrid_rag.retrieval.rrf import reciprocal_rank_fusion
from hybrid_rag.retrieval.vector_retriever import VectorRetriever, _base_node_id

# ─── QueryAnalyzer ─────────────────────────────────────────────────────────────


class TestQueryAnalyzer:
    def test_local_queries(self):
        # All local codebase queries should resolve to "local" query type
        assert analyze("which classes inherit from BaseEmbedder?").query_type == "local"
        assert analyze("explain how the embedding pipeline works").query_type == "local"
        assert analyze("explain how FalkorDB stores the inheritance graph").query_type == "local"
        assert analyze("codebase structure overview").query_type == "local"
        assert analyze("which functions call parse_repo?").query_type == "local"
        assert analyze("what does GraphStore import?").query_type == "local"
        assert analyze("describe how chunking works").query_type == "local"

    def test_global_queries(self):
        # Global architecture queries should resolve to "global" query type
        assert analyze("summarize the codebase").query_type == "global"
        assert analyze("repository summary").query_type == "global"
        assert analyze("architecture design").query_type == "global"
        assert analyze("tổng quan kiến trúc dự án").query_type == "global"
        assert analyze("tóm tắt cấu trúc codebase").query_type == "global"

    def test_entity_extraction_pascal(self):
        a = analyze("what does BaseEmbedder do?")
        assert "BaseEmbedder" in a.entities

    def test_entity_extraction_multi_pascal(self):
        a = analyze("how does OllamaEmbedder extend BaseEmbedder?")
        assert "OllamaEmbedder" in a.entities
        assert "BaseEmbedder" in a.entities

    def test_entity_extraction_snake_case(self):
        a = analyze("show me parse_repo usage")
        assert "parse_repo" in a.entities

    def test_entity_extraction_backtick(self):
        a = analyze("what is the purpose of `OllamaEmbedder`?")
        assert "OllamaEmbedder" in a.entities

    def test_entity_extraction_quoted(self):
        a = analyze('where is "FalkorDBStore" defined?')
        assert "FalkorDBStore" in a.entities

    def test_no_entities_returns_keywords(self):
        a = analyze("explain the embedding process in detail")
        assert len(a.keywords) > 0

    def test_entities_deduplicated(self):
        a = analyze("BaseEmbedder inherits from BaseEmbedder")
        assert a.entities.count("BaseEmbedder") == 1

    def test_keywords_exclude_stop_words(self):
        a = analyze("what is the purpose of the class")
        stop = {"what", "the", "is", "of", "class"}
        for kw in a.keywords:
            assert kw not in stop

    def test_vietnamese_query_analysis(self):
        a = analyze("mô tả về luồng của graph db đi")
        assert a.query_type == "local"
        assert "luồng" in a.keywords
        assert "graph" in a.keywords
        assert "db" in a.keywords
        assert "về" not in a.keywords
        assert "của" not in a.keywords
        assert "mô" not in a.keywords

    def test_query_analysis_repr(self):
        a = QueryAnalysis(query_type="local", entities=["Foo"], keywords=["bar"])
        assert "local" in repr(a)
        assert "Foo" in repr(a)


# ─── RRF ──────────────────────────────────────────────────────────────────────


class TestRRF:
    def _item(self, node_id: str, text: str = "", source: str = "test") -> dict:
        return {"base_node_id": node_id, "node_id": node_id, "text": text, "source": source}

    def test_single_list_preserves_relative_order(self):
        lst = [self._item("a"), self._item("b"), self._item("c")]
        result = reciprocal_rank_fusion(lst)
        ids = [r["base_node_id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_higher_ranked_in_both_lists_wins(self):
        list1 = [self._item("x"), self._item("y")]
        list2 = [self._item("x"), self._item("z")]
        result = reciprocal_rank_fusion(list1, list2)
        assert result[0]["base_node_id"] == "x"

    def test_rrf_score_formula_k60_rank1(self):
        # 1 / (60 + 1) = 1/61
        lst = [self._item("doc")]
        result = reciprocal_rank_fusion(lst, k=60)
        assert abs(result[0]["rrf_score"] - 1 / 61) < 1e-10

    def test_scores_sum_across_two_lists(self):
        # rank-1 in both lists → 1/61 + 1/61
        lst1 = [self._item("doc")]
        lst2 = [self._item("doc")]
        result = reciprocal_rank_fusion(lst1, lst2, k=60)
        assert abs(result[0]["rrf_score"] - 2 / 61) < 1e-10

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_text_merged_from_vector(self):
        graph = [{"base_node_id": "n1", "node_id": "n1", "text": "", "source": "graph"}]
        vector = [{"base_node_id": "n1", "node_id": "n1", "text": "some code", "source": "vector"}]
        result = reciprocal_rank_fusion(graph, vector)
        assert result[0]["text"] == "some code"

    def test_source_set_to_hybrid_when_merged(self):
        graph = [{"base_node_id": "n1", "node_id": "n1", "text": "", "source": "graph"}]
        vector = [{"base_node_id": "n1", "node_id": "n1", "text": "x", "source": "vector"}]
        result = reciprocal_rank_fusion(graph, vector)
        assert result[0]["source"] == "hybrid"

    def test_skips_items_without_id(self):
        lst = [{"text": "no id here", "source": "x"}]
        result = reciprocal_rank_fusion(lst)
        assert result == []

    def test_custom_id_key(self):
        lst = [{"my_id": "a", "text": "t", "source": "s"}]
        result = reciprocal_rank_fusion(lst, id_key="my_id")
        assert result[0]["my_id"] == "a"

    def test_result_has_rrf_score_key(self):
        lst = [self._item("x")]
        result = reciprocal_rank_fusion(lst)
        assert "rrf_score" in result[0]

    def test_three_lists_ordering(self):
        # "shared" is rank-1 in all three → highest score
        l1 = [self._item("shared"), self._item("a")]
        l2 = [self._item("shared"), self._item("b")]
        l3 = [self._item("shared"), self._item("c")]
        result = reciprocal_rank_fusion(l1, l2, l3)
        assert result[0]["base_node_id"] == "shared"


# ─── ContextAssembler ─────────────────────────────────────────────────────────


class TestContextAssembler:
    def _result(
        self,
        node_id: str = "n1",
        text: str = "def foo(): pass",
        source: str = "vector",
        file_path: str = "src/foo.py",
        label: str = "Function",
        name: str = "foo",
    ) -> dict:
        return {
            "node_id": node_id,
            "base_node_id": node_id,
            "name": name,
            "label": label,
            "file_path": file_path,
            "text": text,
            "source": source,
            "rrf_score": 0.02,
            "rel": "",
        }

    def test_returns_retrieval_context(self):
        ctx = ContextAssembler().assemble([self._result()])
        assert isinstance(ctx, RetrievalContext)

    def test_text_contains_file_path(self):
        ctx = ContextAssembler().assemble([self._result(file_path="src/foo.py")])
        assert "src/foo.py" in ctx.text

    def test_text_contains_code(self):
        ctx = ContextAssembler().assemble([self._result(text="def foo(): pass")])
        assert "def foo(): pass" in ctx.text

    def test_top_n_limits_output(self):
        items = [self._result(node_id=str(i)) for i in range(10)]
        ctx = ContextAssembler().assemble(items, top_n=3)
        assert ctx.metadata["shown"] == 3
        assert ctx.metadata["total_results"] == 10

    def test_graph_only_shows_structural_annotation(self):
        item = self._result(text="", source="graph")
        ctx = ContextAssembler().assemble([item])
        assert "Structural node" in ctx.text

    def test_metadata_has_graph_flag(self):
        items = [self._result(source="vector"), self._result(node_id="n2", source="graph")]
        ctx = ContextAssembler().assemble(items, top_n=2)
        assert ctx.metadata["has_graph"] is True
        assert ctx.metadata["has_vector"] is True

    def test_hybrid_source_counted(self):
        item = self._result(source="hybrid")
        ctx = ContextAssembler().assemble([item])
        assert ctx.metadata["has_graph"] is True
        assert ctx.metadata["has_vector"] is True

    def test_query_included_in_text(self):
        ctx = ContextAssembler().assemble([], query="what is BaseEmbedder?")
        assert "what is BaseEmbedder?" in ctx.text

    def test_empty_results(self):
        ctx = ContextAssembler().assemble([])
        assert isinstance(ctx, RetrievalContext)
        assert ctx.metadata["total_results"] == 0
        assert ctx.metadata["shown"] == 0

    def test_chunks_contains_selected(self):
        items = [self._result(node_id=str(i)) for i in range(5)]
        ctx = ContextAssembler().assemble(items, top_n=3)
        assert len(ctx.chunks) == 3


# ─── GraphRetriever ────────────────────────────────────────────────────────────


class TestGraphRetriever:
    def _store(self, nodes=None, neighbors=None):
        store = MagicMock()
        store.find_nodes.return_value = nodes or []
        store.find_neighbors.return_value = neighbors or []
        return store

    def _analysis(self, query_type="structural", entities=None, keywords=None):
        return QueryAnalysis(
            query_type=query_type,
            entities=entities or [],
            keywords=keywords or [],
        )

    def _node(self, nid: str, name: str = "Foo", label: str = "Class") -> dict:
        return {"node_id": nid, "name": name, "label": label, "file_path": "f.py"}

    def test_empty_when_no_terms(self):
        retriever = GraphRetriever(self._store())
        assert retriever.retrieve(self._analysis(entities=[], keywords=[])) == []

    def test_calls_find_nodes_for_entity(self):
        store = self._store(nodes=[self._node("n1", "BaseEmbedder")])
        analysis = self._analysis(entities=["BaseEmbedder"])
        results = GraphRetriever(store).retrieve(analysis)
        store.find_nodes.assert_called_once_with("BaseEmbedder", limit=20)
        assert results[0]["node_id"] == "n1"

    def test_result_source_is_graph(self):
        store = self._store(nodes=[self._node("n1")])
        results = GraphRetriever(store).retrieve(self._analysis(entities=["Foo"]))
        assert results[0]["source"] == "graph"

    def test_expands_neighbors_for_structural(self):
        store = self._store(
            nodes=[self._node("n1", "Base")],
            neighbors=[
                {
                    "src_id": "n1",
                    "rel": "INHERITS",
                    "dst_id": "n2",
                    "dst_name": "Child",
                    "dst_label": "Class",
                    "dst_file_path": "g.py",
                }
            ],
        )
        results = GraphRetriever(store).retrieve(self._analysis(entities=["Base"]))
        ids = {r["node_id"] for r in results}
        assert "n1" in ids
        assert "n2" in ids

    def test_no_neighbor_expansion_for_global(self):
        store = self._store(nodes=[self._node("n1")])
        GraphRetriever(store).retrieve(self._analysis(query_type="global", entities=["Foo"]))
        store.find_neighbors.assert_not_called()

    def test_deduplicates_across_terms(self):
        node = self._node("n1", "Foo")
        store = self._store()
        store.find_nodes.side_effect = [[node], [node]]  # same node for 2 search terms
        analysis = self._analysis(entities=["Foo", "FooAlias"])
        results = GraphRetriever(store).retrieve(analysis)
        assert sum(1 for r in results if r["node_id"] == "n1") == 1

    def test_keywords_searched_after_entities(self):
        store = self._store()
        store.find_nodes.return_value = []
        analysis = self._analysis(entities=["Foo"], keywords=["bar"])
        GraphRetriever(store).retrieve(analysis)
        calls = [c.args[0] for c in store.find_nodes.call_args_list]
        # Graph search uses entities only; keywords belong to vector search
        # to avoid noisy structural matches on generic terms like "inherit".
        assert "Foo" in calls
        assert "bar" not in calls

    def test_top_k_respected(self):
        nodes = [self._node(f"n{i}", f"Node{i}") for i in range(50)]
        store = self._store(nodes=nodes)
        results = GraphRetriever(store).retrieve(self._analysis(entities=["Node"]), top_k=10)
        assert len(results) <= 10


# ─── VectorRetriever ──────────────────────────────────────────────────────────


class TestVectorRetriever:
    def _store(self, hits=None):
        store = MagicMock()
        store.search.return_value = hits or []
        return store

    def _embedder(self, vec=None):
        emb = MagicMock()
        emb.embed_query.return_value = vec or [0.1] * 768
        return emb

    def _hit(self, nid: str, text: str = "code", score: float = 0.9) -> dict:
        return {
            "node_id": nid,
            "label": "Function",
            "file_path": "f.py",
            "text": text,
            "score": score,
        }

    def test_embeds_query_and_searches(self):
        store = self._store([self._hit("n1::0")])
        embedder = self._embedder()
        results = VectorRetriever(store, embedder).retrieve("what does foo do?")
        embedder.embed_query.assert_called_once_with("what does foo do?")
        store.search.assert_called_once()
        assert len(results) == 1

    def test_source_is_vector(self):
        results = VectorRetriever(self._store([self._hit("n1::0")]), self._embedder()).retrieve("q")
        assert results[0]["source"] == "vector"

    def test_base_node_id_strips_chunk_index(self):
        results = VectorRetriever(
            self._store([self._hit("math_utils.py::MyClass::add::0")]),
            self._embedder(),
        ).retrieve("q")
        assert results[0]["base_node_id"] == "math_utils.py::MyClass::add"

    def test_base_node_id_strips_multi_digit_index(self):
        results = VectorRetriever(
            self._store([self._hit("path::func::15")]),
            self._embedder(),
        ).retrieve("q")
        assert results[0]["base_node_id"] == "path::func"

    def test_base_node_id_no_chunk_index(self):
        assert _base_node_id("module::func") == "module::func"
        assert _base_node_id("plain") == "plain"

    def test_base_node_id_digit_suffix(self):
        assert _base_node_id("module::func::0") == "module::func"
        assert _base_node_id("module::func::15") == "module::func"

    def test_empty_hits_returns_empty(self):
        results = VectorRetriever(self._store([]), self._embedder()).retrieve("q")
        assert results == []

    def test_passes_filter_payload(self):
        store = self._store([])
        VectorRetriever(store, self._embedder()).retrieve("q", filter_payload={"label": "Class"})
        _, kwargs = store.search.call_args
        assert kwargs.get("filter_payload") == {"label": "Class"}

    def test_passes_top_k_to_store(self):
        store = self._store([])
        VectorRetriever(store, self._embedder()).retrieve("q", top_k=7)
        _, kwargs = store.search.call_args
        assert kwargs.get("top_k") == 7


# ─── HybridRetriever ─────────────────────────────────────────────────────────


class TestHybridRetriever:
    def _retriever(self, graph_nodes=None, graph_neighbors=None, vector_hits=None):
        graph_store = MagicMock()
        graph_store.find_nodes.return_value = graph_nodes or []
        graph_store.find_neighbors.return_value = graph_neighbors or []
        vector_store = MagicMock()
        vector_store.search.return_value = vector_hits or []
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 768
        return HybridRetriever(
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
        )

    def _hit(self, nid: str, text: str = "code", score: float = 0.9) -> dict:
        return {"node_id": nid, "label": "Class", "file_path": "f.py", "text": text, "score": score}

    def test_returns_list(self):
        retriever = self._retriever(vector_hits=[self._hit("n1::0")])
        assert isinstance(retriever.retrieve("what is BaseEmbedder?"), list)

    def test_results_have_rrf_score(self):
        retriever = self._retriever(vector_hits=[self._hit("n1::0")])
        for r in retriever.retrieve("explain GraphStore"):
            assert "rrf_score" in r

    def test_top_k_limits_results(self):
        hits = [self._hit(f"n{i}::0", score=0.9 - i * 0.01) for i in range(20)]
        retriever = self._retriever(vector_hits=hits)
        assert len(retriever.retrieve("foo", top_k=5)) <= 5

    def test_global_query_scopes_communities_by_repository(self):
        graph_store = MagicMock()
        graph_store.query.return_value.result_set = [
            ["community-1", "Core", "Core summary"]
        ]
        retriever = HybridRetriever(
            graph_store=graph_store,
            vector_store=MagicMock(),
            embedder=MagicMock(),
        )

        results = retriever.retrieve(
            "summarize the codebase",
            repository="sample-repo",
        )

        cypher, params = graph_store.query.call_args.args
        assert "n.repository = $repository" in cypher
        assert params == {"repository": "sample-repo"}
        assert results[0]["name"] == "Core"

    def test_retrieve_with_context_type(self):
        retriever = self._retriever(vector_hits=[self._hit("n1::0")])
        ctx = retriever.retrieve_with_context("how does parse_repo work?")
        assert isinstance(ctx, RetrievalContext)
        assert ctx.text

    def test_graph_used_for_structural_query(self):
        graph_store = MagicMock()
        graph_store.find_nodes.return_value = [
            {"node_id": "n1", "name": "BaseEmbedder", "label": "Class", "file_path": "f.py"}
        ]
        graph_store.find_neighbors.return_value = []
        vector_store = MagicMock()
        vector_store.search.return_value = []
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 768
        retriever = HybridRetriever(
            graph_store=graph_store, vector_store=vector_store, embedder=embedder
        )
        retriever.retrieve("which classes inherit from BaseEmbedder?")
        graph_store.find_nodes.assert_called()

    def test_graph_not_called_for_pure_semantic(self):
        graph_store = MagicMock()
        graph_store.find_nodes.return_value = []
        graph_store.find_neighbors.return_value = []
        vector_store = MagicMock()
        vector_store.search.return_value = [self._hit("n1::0")]
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 768
        retriever = HybridRetriever(
            graph_store=graph_store, vector_store=vector_store, embedder=embedder
        )
        # "explain" + "how works" are semantic-only signals without entity triggers
        retriever.retrieve("explain how the algorithm works")
        # For pure semantic, graph store should NOT be called (query_type == "semantic")
        graph_store.find_nodes.assert_not_called()

    def test_close_calls_embedder_close(self):
        graph_store = MagicMock()
        graph_store.find_nodes.return_value = []
        graph_store.find_neighbors.return_value = []
        vector_store = MagicMock()
        vector_store.search.return_value = []
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 768
        retriever = HybridRetriever(
            graph_store=graph_store, vector_store=vector_store, embedder=embedder
        )
        retriever.close()
        embedder.close.assert_called_once()

    def test_context_manager(self):
        graph_store = MagicMock()
        graph_store.find_nodes.return_value = []
        graph_store.find_neighbors.return_value = []
        vector_store = MagicMock()
        vector_store.search.return_value = []
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 768
        with HybridRetriever(graph_store=graph_store, vector_store=vector_store, embedder=embedder):
            pass
        embedder.close.assert_called_once()

    def test_is_vietnamese_detection(self):
        from hybrid_rag.api.main import _is_vietnamese

        assert _is_vietnamese("mô tả về luồng của graph db đi") is True
        assert _is_vietnamese("mo ta luong graph va cach tong hop") is True
        assert _is_vietnamese("how does the graph database work?") is False
        assert _is_vietnamese("explain this code") is False

    def test_timings_populated(self):
        retriever = self._retriever(vector_hits=[self._hit("n1::0")])
        ctx = retriever.retrieve_with_context("how does parse_repo work?")
        assert "parse_query_ms" in ctx.timings
        assert "graph_search_ms" in ctx.timings
        assert "vector_search_ms" in ctx.timings
        assert "rrf_ms" in ctx.timings
        assert ctx.timings["parse_query_ms"] >= 0.0
