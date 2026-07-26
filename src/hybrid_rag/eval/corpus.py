"""
Evaluation corpus — Q1-Q50 diagnostic queries for the LlamaIndex codebase.

Each QueryCase defines:
  - question: natural language query
  - hops: minimum graph edge traversals needed for a correct answer
  - query_type: how the hybrid system should classify it
  - ground_truth_cypher: Cypher executed against FalkorDB at eval time;
    must RETURN a column whose values are matched against retrieved node names
  - gt_col_index: which column index (0-based) in the Cypher result holds the
    expected name strings

Ground truth is computed *dynamically* from the live graph, so the corpus stays
correct regardless of which repository is indexed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryCase:
    id: str
    question: str
    hops: int
    query_type: str  # "structural" | "hybrid" | "semantic"
    ground_truth_cypher: str
    gt_col_index: int = 0  # column index in Cypher result for expected name
    notes: str = ""


@dataclass(frozen=True)
class RepoQACase:
    id: str
    question: str  # Natural language function behavior description
    target_function: str  # Name of function (e.g. "add")
    file_path: str  # File containing the function (e.g. "math_utils.py")
    notes: str = ""


def load_repoqa_json(filepath: Path | str) -> list[RepoQACase]:
    """Load RepoQA cases from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"RepoQA benchmark file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for item in data:
        cases.append(
            RepoQACase(
                id=item["id"],
                question=item["question"],
                target_function=item["target_function"],
                file_path=item["file_path"],
                notes=item.get("notes", ""),
            )
        )
    return cases


# ── Q1-Q5: 1-hop ──────────────────────────────────────────────────────────────

Q1 = QueryCase(
    id="Q1",
    question="What methods does the BaseRetriever class define?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'BaseRetriever', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="1-hop DEFINES; large class with many methods",
)

Q2 = QueryCase(
    id="Q2",
    question="What modules does llama_index/core/query_engine/retriever_query_engine.py import?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(i) "
        "WHERE m.repository = $repository "
        "AND m.file_path CONTAINS 'retriever_query_engine' "
        "RETURN i.name"
    ),
    notes="1-hop IMPORTS; file-level import lookup",
)

Q3 = QueryCase(
    id="Q3",
    question="Which functions directly call retrieve()?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name = 'retrieve' OR t.id = '__call__retrieve') "
        "RETURN f.name"
    ),
    notes="1-hop CALLS; receiver type is intentionally not inferred",
)

Q4 = QueryCase(
    id="Q4",
    question="Which classes directly inherit from BaseSynthesizer?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseSynthesizer', repository:$repository}) RETURN c.name"
    ),
    notes="1-hop INHERITS; expected 7 subclasses",
)

Q5 = QueryCase(
    id="Q5",
    question="Which methods are defined on VectorIndexRetriever, where similarity_top_k is configured?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'VectorIndexRetriever', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="1-hop DEFINES; all methods of VectorIndexRetriever are relevant",
)

# ── Q6-Q10: 2-hop ─────────────────────────────────────────────────────────────

Q6 = QueryCase(
    id="Q6",
    question="What methods does BaseIndex define for its direct subclasses to inherit?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (sub:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseIndex', repository:$repository})-[:DEFINES]->(f:Function) "
        "RETURN f.name"
    ),
    notes="2-hop INHERITS+DEFINES; methods inherited by all BaseIndex subclasses",
)

Q7 = QueryCase(
    id="Q7",
    question="Which functions call embed_model.get_text_embedding()?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name = 'get_text_embedding' OR t.id = '__call__get_text_embedding') "
        "RETURN f.name"
    ),
    notes="2-hop CALLS; callers of the embedding function",
)

Q8 = QueryCase(
    id="Q8",
    question="What modules occur two import hops downstream of query_engine modules?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(mid:Module)-[:IMPORTS]->(dep:Module) "
        "WHERE m.repository = $repository AND m.name CONTAINS 'query_engine' "
        "RETURN dep.name"
    ),
    notes="2-hop IMPORTS chain; transitive deps of query_engine modules",
)

Q9 = QueryCase(
    id="Q9",
    question="Which subclasses of BaseRetriever override the _retrieve method?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseRetriever', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(f:Function {name:'_retrieve'}) "
        "RETURN c.name"
    ),
    notes="2-hop INHERITS+DEFINES; subclasses that override _retrieve",
)

Q10 = QueryCase(
    id="Q10",
    question="Which modules directly depend on `storage_context`?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(sc:Module) "
        "WHERE m.repository = $repository "
        "AND (sc.name CONTAINS 'storage_context' OR sc.file_path CONTAINS 'storage_context') "
        "RETURN m.name"
    ),
    notes="Direct incoming IMPORTS edges for the storage_context module",
)

# ── Q11-Q15: 3-hop ────────────────────────────────────────────────────────────

Q11 = QueryCase(
    id="Q11",
    question="Which functions can reach retrieve() within three calls?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS*1..3]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name = 'retrieve' OR t.id = '__call__retrieve') "
        "RETURN f.name"
    ),
    notes="3-hop CALLS reachability without unsupported receiver-type inference",
)

Q12 = QueryCase(
    id="Q12",
    question="Trace the ancestor hierarchy of RetrieverQueryEngine.",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'RetrieverQueryEngine', repository:$repository})"
        "-[:INHERITS*1..5]->(b:Class) RETURN b.name"
    ),
    notes="3+-hop INHERITS chain; full ancestor hierarchy",
)

Q13 = QueryCase(
    id="Q13",
    question="Which modules transitively depend on the VectorStoreIndex module?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS*1..3]->(dep:Module) "
        "WHERE m.repository = $repository "
        "AND dep.file_path CONTAINS 'indices/vector_store' "
        "RETURN m.name"
    ),
    notes="3-hop IMPORTS chain supported by wheel source without requiring tests",
)

Q14 = QueryCase(
    id="Q14",
    question="Which function is the entry-point anchor for index.as_query_engine()?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function {name:'as_query_engine', repository:$repository}) RETURN f.name"
    ),
    notes="Entry-point anchor for call-chain tracing",
)

Q15 = QueryCase(
    id="Q15",
    question="Which NodeWithScore methods form the class-level refactoring surface?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'NodeWithScore', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="NodeWithScore methods as blast radius anchor",
)

# ── Q16-Q20: hybrid / semantic ────────────────────────────────────────────────

Q16 = QueryCase(
    id="Q16",
    question="Which classes have retry or fallback in their names?",
    hops=2,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (c:Class) "
        "WHERE c.repository = $repository "
        "AND (c.name CONTAINS 'Retry' OR c.name CONTAINS 'Fallback' "
        "OR c.name CONTAINS 'retry' OR c.name CONTAINS 'fallback') "
        "RETURN c.name"
    ),
    notes="Hybrid: semantic for retry/fallback concept, graph for hierarchy",
)

Q17 = QueryCase(
    id="Q17",
    question="Which modules are named for events, callbacks, hooks, or dispatchers?",
    hops=2,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (m:Module) "
        "WHERE m.repository = $repository "
        "AND (m.name CONTAINS 'event' OR m.name CONTAINS 'callback' "
        "OR m.name CONTAINS 'hook' OR m.name CONTAINS 'dispatcher') "
        "RETURN m.name"
    ),
    notes="Hybrid: semantic for pattern, graph for module structure",
)

Q18 = QueryCase(
    id="Q18",
    question="Which async retrieve, query, embed, or generate functions use the conventional a-prefix?",
    hops=2,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (f:Function) "
        "WHERE f.repository = $repository AND f.name STARTS WITH 'a' "
        "AND (f.name CONTAINS 'retrieve' OR f.name CONTAINS 'query' "
        "OR f.name CONTAINS 'embed' OR f.name CONTAINS 'generate') "
        "RETURN f.name"
    ),
    notes="Hybrid: explicitly measures the async naming convention",
)

Q19 = QueryCase(
    id="Q19",
    question="Which core query-engine, retriever, and synthesizer classes participate in query flow?",
    hops=3,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (c:Class) "
        "WHERE c.repository = $repository "
        "AND c.name IN ['QueryEngine', 'RetrieverQueryEngine', 'BaseRetriever', "
        "'VectorIndexRetriever', 'BaseSynthesizer', 'ResponseSynthesizer'] "
        "RETURN c.name"
    ),
    notes="Hybrid: unordered membership in the core query pipeline",
)

Q20 = QueryCase(
    id="Q20",
    question="Which functions are named for reading, writing, loading, saving, or persistence?",
    hops=3,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (f:Function) "
        "WHERE f.repository = $repository "
        "AND (f.name CONTAINS 'read' OR f.name CONTAINS 'write' "
        "OR f.name CONTAINS 'load' OR f.name CONTAINS 'save' "
        "OR f.name CONTAINS 'persist') "
        "RETURN f.name"
    ),
    notes="Hybrid: file I/O function names as proxy for FS access",
)

# ── Q21-Q25: additional 1-hop structural coverage ─────────────────────────────

Q21 = QueryCase(
    id="Q21",
    question="List the methods defined directly by BaseEmbedding.",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'BaseEmbedding', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="Paraphrase coverage for direct DEFINES traversal",
)

Q22 = QueryCase(
    id="Q22",
    question="Which classes are immediate subclasses of BaseQueryEngine?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseQueryEngine', repository:$repository}) RETURN c.name"
    ),
    notes="Incoming one-hop INHERITS traversal",
)

Q23 = QueryCase(
    id="Q23",
    question="Which modules directly depend on `settings`?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(d:Module) "
        "WHERE m.repository = $repository AND d.name = 'settings' RETURN m.name"
    ),
    notes="Incoming one-hop IMPORTS traversal",
)

Q24 = QueryCase(
    id="Q24",
    question="Which functions directly invoke get_content()?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name = 'get_content' OR t.id = '__call__get_content') RETURN f.name"
    ),
    notes="Callable extraction and incoming one-hop CALLS traversal",
)

Q25 = QueryCase(
    id="Q25",
    question="What methods does SimpleDirectoryReader define directly?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'SimpleDirectoryReader', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="Second direct DEFINES paraphrase",
)

# ── Q26-Q35: additional 2-hop and composed structural coverage ────────────────

Q26 = QueryCase(
    id="Q26",
    question="Which BaseQueryEngine subclasses define their own methods?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseQueryEngine', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(:Function) RETURN DISTINCT c.name"
    ),
    notes="Composed INHERITS plus DEFINES traversal",
)

Q27 = QueryCase(
    id="Q27",
    question="Which subclasses of BaseNodePostprocessor override _postprocess_nodes()?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseNodePostprocessor', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(:Function {name:'_postprocess_nodes'}) RETURN c.name"
    ),
    notes="Override query requiring INHERITS plus DEFINES",
)

Q28 = QueryCase(
    id="Q28",
    question="Which subclasses of BaseEmbedding implement _get_query_embedding()?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseEmbedding', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(:Function {name:'_get_query_embedding'}) RETURN c.name"
    ),
    notes="Embedding override query requiring two relation types",
)

Q29 = QueryCase(
    id="Q29",
    question="Which subclasses of BaseRetriever provide _retrieve()?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseRetriever', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(:Function {name:'_retrieve'}) RETURN c.name"
    ),
    notes="Paraphrase of override intent for robustness",
)

Q30 = QueryCase(
    id="Q30",
    question="Which BaseSynthesizer subclasses define synthesize()?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseSynthesizer', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(:Function {name:'synthesize'}) RETURN c.name"
    ),
    notes="Synthesizer override query requiring two relation types",
)

Q31 = QueryCase(
    id="Q31",
    question="Which modules depend on `storage_context` within two import hops?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS*1..2]->(d:Module) "
        "WHERE m.repository = $repository "
        "AND (d.name CONTAINS 'storage_context' OR d.file_path CONTAINS 'storage_context') "
        "RETURN m.name"
    ),
    notes="Bounded incoming transitive IMPORTS traversal",
)

Q32 = QueryCase(
    id="Q32",
    question="Which modules depend on base_query_engine within two import hops?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS*1..2]->(d:Module) "
        "WHERE m.repository = $repository "
        "AND (d.name = 'base_query_engine' OR d.file_path CONTAINS 'base_query_engine') "
        "RETURN m.name"
    ),
    notes="Dependency wording with explicit two-hop bound",
)

Q33 = QueryCase(
    id="Q33",
    question="Which functions can reach get_content() within two calls?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS*1..2]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name = 'get_content' OR t.id = '__call__get_content') RETURN f.name"
    ),
    notes="Bounded reverse CALLS traversal",
)

Q34 = QueryCase(
    id="Q34",
    question="What methods does VectorStoreIndex inherit from its direct base classes?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'VectorStoreIndex', repository:$repository})"
        "-[:INHERITS]->(:Class)-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="Composed outgoing INHERITS plus DEFINES traversal",
)

Q35 = QueryCase(
    id="Q35",
    question="What modules are imported by retriever_query_engine.py within two import hops?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(:Module)-[:IMPORTS]->(d:Module) "
        "WHERE m.repository = $repository "
        "AND m.file_path CONTAINS 'retriever_query_engine' RETURN d.name"
    ),
    notes="Exact two-hop outgoing IMPORTS traversal",
)

# ── Q36-Q40: additional 3-hop structural coverage ─────────────────────────────

Q36 = QueryCase(
    id="Q36",
    question="Which functions can reach get_response() within three calls?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS*1..3]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name = 'get_response' OR t.id = '__call__get_response') RETURN f.name"
    ),
    notes="Second bounded reverse CALLS query",
)

Q37 = QueryCase(
    id="Q37",
    question="Trace the ancestor hierarchy of AgentWorkflow up to three levels.",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'AgentWorkflow', repository:$repository})"
        "-[:INHERITS*1..3]->(b:Class) RETURN b.name"
    ),
    notes="Bounded multi-hop inheritance traversal",
)

Q38 = QueryCase(
    id="Q38",
    question="Which modules transitively depend on async_utils within three imports?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS*1..3]->(d:Module) "
        "WHERE m.repository = $repository "
        "AND (d.name = 'async_utils' OR d.file_path CONTAINS 'async_utils') RETURN m.name"
    ),
    notes="Bounded incoming three-hop IMPORTS traversal",
)

Q39 = QueryCase(
    id="Q39",
    question="Which functions are reachable from as_query_engine() within three calls?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function {name:'as_query_engine', repository:$repository})"
        "-[:CALLS*1..3]->(t:Function) RETURN t.name"
    ),
    notes="Bounded outgoing CALLS traversal from a public entry point",
)

Q40 = QueryCase(
    id="Q40",
    question="Which functions are called by query() within three calls?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function {name:'query', repository:$repository})"
        "-[:CALLS*1..3]->(t:Function) RETURN t.name"
    ),
    notes="Ambiguous callable name with bounded outgoing traversal",
)

# ── Q41-Q45: additional semantic and hybrid coverage ──────────────────────────

Q41 = QueryCase(
    id="Q41",
    question="Which classes represent streaming behavior?",
    hops=1,
    query_type="semantic",
    ground_truth_cypher=(
        "MATCH (c:Class) WHERE c.repository = $repository "
        "AND (c.name CONTAINS 'Stream' OR c.name CONTAINS 'Streaming') RETURN c.name"
    ),
    notes="Semantic concept measured by explicit class names",
)

Q42 = QueryCase(
    id="Q42",
    question="Which functions implement retry or fallback behavior?",
    hops=1,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (f:Function) WHERE f.repository = $repository "
        "AND (f.name CONTAINS 'retry' OR f.name CONTAINS 'fallback' "
        "OR f.name CONTAINS 'Retry' OR f.name CONTAINS 'Fallback') RETURN f.name"
    ),
    notes="Hybrid concept lookup over function names",
)

Q43 = QueryCase(
    id="Q43",
    question="Which modules are named for callbacks, hooks, events, or dispatchers?",
    hops=1,
    query_type="semantic",
    ground_truth_cypher=(
        "MATCH (m:Module) WHERE m.repository = $repository "
        "AND (m.name CONTAINS 'callback' OR m.name CONTAINS 'hook' "
        "OR m.name CONTAINS 'event' OR m.name CONTAINS 'dispatcher') RETURN m.name"
    ),
    notes="Honest naming-based oracle for observer-related modules",
)

Q44 = QueryCase(
    id="Q44",
    question="Which a-prefixed functions perform asynchronous chat or completion work?",
    hops=1,
    query_type="semantic",
    ground_truth_cypher=(
        "MATCH (f:Function) WHERE f.repository = $repository AND f.name STARTS WITH 'a' "
        "AND (f.name CONTAINS 'chat' OR f.name CONTAINS 'complete') RETURN f.name"
    ),
    notes="Async naming convention for chat and completion functions",
)

Q45 = QueryCase(
    id="Q45",
    question="Which functions are explicitly named for persistence or saving?",
    hops=1,
    query_type="semantic",
    ground_truth_cypher=(
        "MATCH (f:Function) WHERE f.repository = $repository "
        "AND (f.name CONTAINS 'persist' OR f.name CONTAINS 'save') RETURN f.name"
    ),
    notes="Semantic file-persistence proxy with a precise oracle",
)

# ── Q46-Q50: ambiguity and repository-scoping coverage ────────────────────────

Q46 = QueryCase(
    id="Q46",
    question="For the class BaseRetriever—not similarly named modules—which methods does it define?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'BaseRetriever', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="Entity-type disambiguation between classes and modules",
)

Q47 = QueryCase(
    id="Q47",
    question="Which direct subclasses of BaseRetriever override the private _retrieve method?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->"
        "(b:Class {name:'BaseRetriever', repository:$repository}) "
        "MATCH (c)-[:DEFINES]->(:Function {name:'_retrieve'}) RETURN c.name"
    ),
    notes="Public class versus private method disambiguation",
)

Q48 = QueryCase(
    id="Q48",
    question="Which methods are defined on the StorageContext class itself?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'StorageContext', repository:$repository})"
        "-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="Class versus storage_context module disambiguation",
)

Q49 = QueryCase(
    id="Q49",
    question="Which functions call either public retrieve() or private _retrieve() directly?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function) "
        "WHERE f.repository = $repository "
        "AND (t.name IN ['retrieve', '_retrieve'] "
        "OR t.id IN ['__call__retrieve', '__call___retrieve']) RETURN f.name"
    ),
    notes="Disambiguates public and private callable spellings",
)

Q50 = QueryCase(
    id="Q50",
    question="Which modules in this repository directly depend on base_retriever?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(d:Module) "
        "WHERE m.repository = $repository "
        "AND (d.name = 'base_retriever' OR d.file_path CONTAINS 'base_retriever') "
        "RETURN m.name"
    ),
    notes="Repository-scoped module-name disambiguation",
)


# ── Corpus registry ───────────────────────────────────────────────────────────

EVAL_CORPUS: list[QueryCase] = [
    Q1,
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q7,
    Q8,
    Q9,
    Q10,
    Q11,
    Q12,
    Q13,
    Q14,
    Q15,
    Q16,
    Q17,
    Q18,
    Q19,
    Q20,
    Q21,
    Q22,
    Q23,
    Q24,
    Q25,
    Q26,
    Q27,
    Q28,
    Q29,
    Q30,
    Q31,
    Q32,
    Q33,
    Q34,
    Q35,
    Q36,
    Q37,
    Q38,
    Q39,
    Q40,
    Q41,
    Q42,
    Q43,
    Q44,
    Q45,
    Q46,
    Q47,
    Q48,
    Q49,
    Q50,
]

ONE_HOP = [Q1, Q2, Q3, Q4, Q5, Q21, Q22, Q23, Q24, Q25]
TWO_HOP = [Q6, Q7, Q8, Q9, Q10, Q26, Q27, Q28, Q29, Q30, Q31, Q32, Q33, Q34, Q35]
THREE_HOP = [Q11, Q12, Q13, Q14, Q15, Q36, Q37, Q38, Q39, Q40]
HYBRID = [Q16, Q17, Q18, Q19, Q20, Q41, Q42, Q43, Q44, Q45]
AMBIGUOUS = [Q46, Q47, Q48, Q49, Q50]
