"""
Evaluation corpus — Q1-Q20 multi-hop queries for LlamaIndex codebase.

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
        "MATCH (c:Class {name:'BaseRetriever'})-[:DEFINES]->(f:Function) RETURN f.name"
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
        "WHERE m.file_path CONTAINS 'retriever_query_engine' "
        "RETURN i.name"
    ),
    notes="1-hop IMPORTS; file-level import lookup",
)

Q3 = QueryCase(
    id="Q3",
    question="Which functions directly call BaseRetriever.retrieve()?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function {name:'retrieve'}) "
        "WHERE t.class_name = 'BaseRetriever' OR t.id CONTAINS '::BaseRetriever::retrieve' "
        "RETURN f.name"
    ),
    notes="1-hop CALLS; may be sparse if CALLS edges absent",
)

Q4 = QueryCase(
    id="Q4",
    question="Which classes directly inherit from BaseSynthesizer?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class)-[:INHERITS]->(b:Class {name:'BaseSynthesizer'}) RETURN c.name"
    ),
    notes="1-hop INHERITS; expected 7 subclasses",
)

Q5 = QueryCase(
    id="Q5",
    question="Where is the similarity_top_k parameter used inside VectorIndexRetriever?",
    hops=1,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'VectorIndexRetriever'})-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="1-hop DEFINES; all methods of VectorIndexRetriever are relevant",
)

# ── Q6-Q10: 2-hop ─────────────────────────────────────────────────────────────

Q6 = QueryCase(
    id="Q6",
    question="What methods are available to all subclasses of BaseIndex through inheritance?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (sub:Class)-[:INHERITS]->(b:Class {name:'BaseIndex'})-[:DEFINES]->(f:Function) "
        "RETURN f.name"
    ),
    notes="2-hop INHERITS+DEFINES; methods inherited by all BaseIndex subclasses",
)

Q7 = QueryCase(
    id="Q7",
    question="Which modules contain functions that call embed_model.get_text_embedding()?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function {name:'get_text_embedding'}) RETURN f.name"
    ),
    notes="2-hop CALLS; callers of the embedding function",
)

Q8 = QueryCase(
    id="Q8",
    question="What does importing QueryEngine transitively bring into the namespace?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(mid:Module)-[:IMPORTS]->(dep:Module) "
        "WHERE m.name CONTAINS 'query_engine' "
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
        "MATCH (c:Class)-[:INHERITS]->(b:Class {name:'BaseRetriever'}) "
        "MATCH (c)-[:DEFINES]->(f:Function {name:'_retrieve'}) "
        "RETURN c.name"
    ),
    notes="2-hop INHERITS+DEFINES; subclasses that override _retrieve",
)

Q10 = QueryCase(
    id="Q10",
    question="Which modules depend on StorageContext and what functions do they use from it?",
    hops=2,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (m:Module)-[:IMPORTS]->(sc:Module) "
        "WHERE sc.name CONTAINS 'storage_context' OR sc.file_path CONTAINS 'storage_context' "
        "RETURN m.name"
    ),
    notes="2-hop IMPORTS; modules directly importing storage_context",
)

# ── Q11-Q15: 3-hop ────────────────────────────────────────────────────────────

Q11 = QueryCase(
    id="Q11",
    question="If I change the signature of BaseRetriever.retrieve(), which functions across which modules are affected?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (f:Function)-[:CALLS]->(t:Function {name:'retrieve'}) "
        "WHERE t.id CONTAINS 'BaseRetriever' "
        "RETURN f.name"
    ),
    notes="3-hop impact analysis via CALLS chain",
)

Q12 = QueryCase(
    id="Q12",
    question="Trace the full method resolution order (MRO) for RetrieverQueryEngine.",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'RetrieverQueryEngine'})-[:INHERITS*1..5]->(b:Class) RETURN b.name"
    ),
    notes="3+-hop INHERITS chain; full ancestor hierarchy",
)

Q13 = QueryCase(
    id="Q13",
    question="Which tests would break if VectorStoreIndex is removed?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (test:Module)-[:IMPORTS*1..3]->(dep:Module) "
        "WHERE test.file_path CONTAINS 'test' "
        "AND (dep.name CONTAINS 'vector_store_index' OR dep.file_path CONTAINS 'vector_store_index') "
        "RETURN test.name"
    ),
    notes="3-hop IMPORTS* chain from test modules to VectorStoreIndex",
)

Q14 = QueryCase(
    id="Q14",
    question="Trace all code paths from user calling index.as_query_engine() to when embeddings are generated.",
    hops=3,
    query_type="structural",
    ground_truth_cypher=("MATCH (f:Function {name:'as_query_engine'}) RETURN f.name"),
    notes="Entry-point anchor for call-chain tracing",
)

Q15 = QueryCase(
    id="Q15",
    question="What is the blast radius of renaming the nodes parameter in NodeWithScore?",
    hops=3,
    query_type="structural",
    ground_truth_cypher=(
        "MATCH (c:Class {name:'NodeWithScore'})-[:DEFINES]->(f:Function) RETURN f.name"
    ),
    notes="NodeWithScore methods as blast radius anchor",
)

# ── Q16-Q20: hybrid / semantic ────────────────────────────────────────────────

Q16 = QueryCase(
    id="Q16",
    question="Find all classes that implement a retry or fallback mechanism and show their inheritance hierarchy.",
    hops=2,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (c:Class) "
        "WHERE c.name CONTAINS 'Retry' OR c.name CONTAINS 'Fallback' "
        "OR c.name CONTAINS 'retry' OR c.name CONTAINS 'fallback' "
        "RETURN c.name"
    ),
    notes="Hybrid: semantic for retry/fallback concept, graph for hierarchy",
)

Q17 = QueryCase(
    id="Q17",
    question="Which modules implement the observer pattern (event callbacks or hooks)?",
    hops=2,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (m:Module) "
        "WHERE m.name CONTAINS 'event' OR m.name CONTAINS 'callback' "
        "OR m.name CONTAINS 'hook' OR m.name CONTAINS 'dispatcher' "
        "RETURN m.name"
    ),
    notes="Hybrid: semantic for pattern, graph for module structure",
)

Q18 = QueryCase(
    id="Q18",
    question="Find all functions documented as async-safe and check if they share a common base class.",
    hops=2,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (f:Function) "
        "WHERE f.name STARTS WITH 'a' AND (f.name CONTAINS 'retrieve' OR f.name CONTAINS 'query' "
        "OR f.name CONTAINS 'embed' OR f.name CONTAINS 'generate') "
        "RETURN f.name"
    ),
    notes="Hybrid: async methods as proxy for async-safe documentation",
)

Q19 = QueryCase(
    id="Q19",
    question="Explain the data flow when a user submits a query: which classes are instantiated and in what order?",
    hops=3,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (c:Class) "
        "WHERE c.name IN ['QueryEngine', 'RetrieverQueryEngine', 'BaseRetriever', "
        "'VectorIndexRetriever', 'BaseSynthesizer', 'ResponseSynthesizer'] "
        "RETURN c.name"
    ),
    notes="Hybrid: query pipeline classes",
)

Q20 = QueryCase(
    id="Q20",
    question="Which functions access the file system directly (open/read/write) and are they reachable from the public API?",
    hops=3,
    query_type="hybrid",
    ground_truth_cypher=(
        "MATCH (f:Function) "
        "WHERE f.name CONTAINS 'read' OR f.name CONTAINS 'write' "
        "OR f.name CONTAINS 'load' OR f.name CONTAINS 'save' "
        "OR f.name CONTAINS 'persist' "
        "RETURN f.name"
    ),
    notes="Hybrid: file I/O function names as proxy for FS access",
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
]

ONE_HOP = [Q1, Q2, Q3, Q4, Q5]
TWO_HOP = [Q6, Q7, Q8, Q9, Q10]
THREE_HOP = [Q11, Q12, Q13, Q14, Q15]
HYBRID = [Q16, Q17, Q18, Q19, Q20]
