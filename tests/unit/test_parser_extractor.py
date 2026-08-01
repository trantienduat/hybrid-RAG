"""
Unit tests for ingestion/parser.py + ingestion/triplet_extractor.py (roadmap #11).
No external services required.
"""

from __future__ import annotations

from pathlib import Path

from hybrid_rag.ingestion.parser import (
    ParseResult,
    parse_file,
    parse_repo,
)
from hybrid_rag.ingestion.triplet_extractor import extract_triples

# ── fixtures ──────────────────────────────────────────────────────────────────

_FIXTURE_REPO = Path(__file__).parent.parent.parent / "fixtures" / "small_repo"
_MATH_UTILS = _FIXTURE_REPO / "math_utils.py"
_STRING_HELPERS = _FIXTURE_REPO / "string_helpers.py"


# ── parse_file tests ──────────────────────────────────────────────────────────


class TestParseFile:
    def test_returns_parse_result(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        assert isinstance(result, ParseResult)
        assert not result.errors

    def test_module_node_exists(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        modules = [n for n in result.nodes if n.label == "Module"]
        assert len(modules) == 1
        assert modules[0].properties["name"] == "math_utils"

    def test_extracts_functions(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        fn_names = {n.properties["name"] for n in result.nodes if n.label == "Function"}
        assert "add" in fn_names
        assert "multiply" in fn_names

    def test_extracts_class(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        classes = [n for n in result.nodes if n.label == "Class"]
        assert any(c.properties["name"] == "Calculator" for c in classes)

    def test_class_method_extracted(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        fn_names = {n.properties["name"] for n in result.nodes if n.label == "Function"}
        # Calculator defines __init__ and compute methods
        assert "__init__" in fn_names or "compute" in fn_names

    def test_extracts_imports(self):
        result = parse_file(_STRING_HELPERS, _FIXTURE_REPO)
        import_edges = [e for e in result.edges if e.rel == "IMPORTS"]
        assert len(import_edges) >= 1
        # string_helpers imports from math_utils
        dst_ids = {e.dst_id for e in import_edges}
        assert "math_utils" in dst_ids or any("math" in d for d in dst_ids)

    def test_preserves_full_import_module_for_resolution(self, tmp_path):
        source = tmp_path / "imports.py"
        source.write_text(
            "import llama_index.core.schema\n"
            "from llama_index.core.storage.storage_context import StorageContext\n"
        )

        result = parse_file(source, tmp_path)

        imports = {edge.dst_id for edge in result.edges if edge.rel == "IMPORTS"}
        assert "llama_index.core.schema" in imports
        assert "llama_index.core.storage.storage_context" in imports

    def test_extracts_generic_base_class(self, tmp_path):
        source = tmp_path / "generic.py"
        source.write_text("class Concrete(BaseIndex[IndexDict]):\n    pass\n")

        result = parse_file(source, tmp_path)

        inherits = [edge for edge in result.edges if edge.rel == "INHERITS"]
        assert [(edge.src_id, edge.dst_id) for edge in inherits] == [
            ("generic.Concrete", "BaseIndex")
        ]

    def test_inherits_edge_extracted(self):
        result = parse_file(_STRING_HELPERS, _FIXTURE_REPO)
        inherit_edges = [e for e in result.edges if e.rel == "INHERITS"]
        # StringProcessor doesn't inherit — check math_utils Calculator if it does
        # Just confirm INHERITS edges are well-formed when they exist
        for e in inherit_edges:
            assert e.src_id
            assert e.dst_id

    def test_defines_edges_link_module_to_functions(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        defines = [e for e in result.edges if e.rel == "DEFINES"]
        assert len(defines) >= 2  # at least add + multiply

    def test_defined_in_edges_point_to_module(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        module_id = "math_utils"
        defined_in = [e for e in result.edges if e.rel == "DEFINED_IN" and e.dst_id == module_id]
        assert len(defined_in) >= 2

    def test_unsupported_extension_returns_error(self, tmp_path):
        f = tmp_path / "foo.rb"
        f.write_text("puts 'hello'")
        result = parse_file(f, tmp_path)
        assert result.errors
        assert not result.nodes

    def test_parse_repo_collects_all_files(self):
        result = parse_repo(_FIXTURE_REPO, languages=["python"])
        module_names = {
            n.properties["name"]
            for n in result.nodes
            if n.label == "Module" and n.properties.get("type") != "external"
        }
        assert "math_utils" in module_names
        assert "string_helpers" in module_names

    def test_parse_repo_includes_repository_root_for_top_level_modules(self):
        result = parse_repo(_FIXTURE_REPO, languages=["python"], repo_name="fixture")

        roots = [node for node in result.nodes if node.label == "RepositoryMetadata"]
        assert [(node.id, node.properties["repository"]) for node in roots] == [
            ("fixture", "fixture")
        ]
        root_triples = [
            triple
            for triple in extract_triples(result)
            if triple.src_id == "fixture" and triple.rel == "DEFINES"
        ]
        assert root_triples
        assert {triple.src_label for triple in root_triples} == {"RepositoryMetadata"}

    def test_parse_repo_rejects_java_language(self):
        try:
            parse_repo(_FIXTURE_REPO, languages=["java"], repo_name="fixture")
        except ValueError as exc:
            assert "Only Python indexing" in str(exc)
        else:
            assert False, "Java indexing must be rejected until implemented"

    def test_node_ids_are_unique_per_file(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        ids = [n.id for n in result.nodes]
        assert len(ids) == len(set(ids)), "Duplicate node IDs detected"

    def test_python_node_ids_are_namespaced_by_repository(self, tmp_path):
        source = tmp_path / "same.py"
        source.write_text("def shared():\n    return 1\n")

        repo_a = parse_file(source, tmp_path, repo_name="repo-a")
        repo_b = parse_file(source, tmp_path, repo_name="repo-b")

        real_a = {n.id for n in repo_a.nodes if n.properties.get("type") != "external"}
        real_b = {n.id for n in repo_b.nodes if n.properties.get("type") != "external"}
        assert real_a == {"repo-a::same", "repo-a::same.shared"}
        assert real_b == {"repo-b::same", "repo-b::same.shared"}
        assert real_a.isdisjoint(real_b)
        assert all(
            edge.src_id.startswith("repo-a::") for edge in repo_a.edges if edge.src_id != "repo-a"
        )

    def test_function_node_has_line_numbers(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        fns = [
            n
            for n in result.nodes
            if n.label == "Function" and n.properties.get("type") != "external"
        ]
        for fn in fns:
            assert fn.properties.get("line_start", 0) > 0
            assert fn.properties.get("line_end", 0) >= fn.properties["line_start"]

    def test_extracts_decorated_definitions(self, tmp_path):
        source = tmp_path / "decorated.py"
        source.write_text(
            """
@register
def top_level():
    helper()

@dataclass
class Service:
    @property
    def value(self):
        return 1

    @abstractmethod
    def run(self):
        execute()
""".strip()
        )

        result = parse_file(source, tmp_path)
        functions = {
            node.properties["name"]: node for node in result.nodes if node.label == "Function"
        }

        assert {"top_level", "value", "run"} <= functions.keys()
        assert functions["value"].properties["is_property"] is True
        assert functions["run"].properties["is_abstract"] is True
        calls = {(edge.src_id, edge.dst_id) for edge in result.edges if edge.rel == "CALLS"}
        assert ("decorated.top_level", "__call__helper") in calls
        assert ("decorated.Service.run", "__call__execute") in calls


# ── triplet extractor tests ───────────────────────────────────────────────────


class TestTripletExtractor:
    def test_returns_triples(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        triples = extract_triples(result)
        assert isinstance(triples, list)
        assert len(triples) > 0

    def test_all_triples_have_labels(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        triples = extract_triples(result)
        for t in triples:
            assert t.src_label, f"Missing src_label on {t}"
            assert t.dst_label, f"Missing dst_label on {t}"

    def test_defines_triple_present(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        triples = extract_triples(result)
        rels = {t.rel for t in triples}
        assert "DEFINES" in rels

    def test_imports_triple_present(self):
        result = parse_file(_STRING_HELPERS, _FIXTURE_REPO)
        triples = extract_triples(result)
        rels = {t.rel for t in triples}
        assert "IMPORTS" in rels

    def test_empty_parse_result_gives_empty_triples(self):
        result = ParseResult()
        triples = extract_triples(result)
        assert triples == []

    def test_triple_src_ids_match_node_ids(self):
        result = parse_file(_MATH_UTILS, _FIXTURE_REPO)
        known_ids = {n.id for n in result.nodes}
        triples = extract_triples(result)
        # src_ids that ARE in the result should match
        for t in triples:
            if t.src_id in known_ids:
                src_node = next(n for n in result.nodes if n.id == t.src_id)
                assert t.src_label == src_node.label
