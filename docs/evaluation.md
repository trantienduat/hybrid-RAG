# Evaluation: Multi-hop Test Queries

Target codebase for queries: **LlamaIndex** (Python, high modularity, deep hierarchies)

The Q1-Q50 corpus is a **diagnostic graph benchmark**. Its ground truth is
computed from repository-scoped Cypher and therefore validates retrieval over
the indexed graph, not the completeness of that graph. A run is rejected if
any case has empty ground truth or if FalkorDB and Qdrant provenance differs.
Use the versioned RepoQA fixtures for an independent source-file retrieval
benchmark.

Each query is tagged with:
- **Hops**: number of graph edge traversals required
- **Path**: expected Cypher traversal pattern
- **Type**: structural / semantic / hybrid
- **Baseline**: can Vector RAG answer this alone?

---

## 1-Hop Queries (direct relationship)

### Q1 — Class members
> "What methods does the `BaseRetriever` class define?"

- Hops: 1
- Path: `(:Class {name:"BaseRetriever"})-[:DEFINES]->(:Function)`
- Type: structural
- Baseline: partial (if chunk contains full class body)

### Q2 — Direct imports
> "What modules does `llama_index/core/query_engine/retriever_query_engine.py` import?"

- Hops: 1
- Path: `(:Module {file_path:"...retriever_query_engine.py"})-[:IMPORTS]->(:Module)`
- Type: structural
- Baseline: partial

### Q3 — Direct callers
> "Which functions directly call `retrieve()`?"

- Hops: 1
- Path: `(:Function)-[:CALLS]->(:Function {name:"retrieve"})`
- Type: structural
- Baseline: weak (semantic search finds description, not call sites)
- Note: receiver-type inference is outside the current static-analysis schema

### Q4 — Inheritance
> "Which classes directly inherit from `BaseSynthesizer`?"

- Hops: 1
- Path: `(:Class)-[:INHERITS]->(:Class {name:"BaseSynthesizer"})`
- Type: structural
- Baseline: very weak

### Q5 — Variable usage
> "Which methods are defined on `VectorIndexRetriever`, where `similarity_top_k` is configured?"

- Path: `(:Class {name:"VectorIndexRetriever"})-[:DEFINES]->(:Function)`
- Type: structural
- Baseline: partial

---

## 2-Hop Queries

### Q6 — Transitive inheritance
> "What methods does `BaseIndex` define for its direct subclasses to inherit?"

- Hops: 2
- Path: `(:Class)-[:INHERITS]->(:Class {name:"BaseIndex"})-[:DEFINES]->(:Function)`
- Type: structural
- Baseline: very weak

### Q7 — Indirect callers
> "Which functions call `embed_model.get_text_embedding()`?"

- Path: `(:Function)-[:CALLS]->(:Function {name:"get_text_embedding"})`
- Type: structural
- Baseline: weak

### Q8 — Import chain
> "What modules occur two import hops downstream of `query_engine` modules?"

- Hops: 2
- Path: `(:Module)-[:IMPORTS]->(:Module)-[:IMPORTS]->(:Module)` filtered by `QueryEngine`
- Type: structural
- Baseline: fails completely

### Q9 — Override detection
> "Which subclasses of `BaseRetriever` override the `_retrieve` method?"

- Hops: 2
- Path: `(:Class)-[:INHERITS]->(:Class {name:"BaseRetriever"})`, then filter subclasses `[:DEFINES]->(:Function {name:"_retrieve"})`
- Type: structural
- Baseline: very weak

### Q10 — Cross-module dependency
> "Which modules directly depend on `storage_context`?"

- Path: `(:Module)-[:IMPORTS]->(:Module {name:"storage_context"})`
- Type: structural
- Baseline: partial

---

## 3-Hop Queries

### Q11 — Impact analysis
> "Which functions can reach `retrieve()` within three calls?"

- Hops: 3
- Path: `(:Function)-[:CALLS*1..3]->(:Function {name:"retrieve"})`
- Type: structural
- Baseline: fails

### Q12 — Deep inheritance chain
> "Trace the ancestor hierarchy of `RetrieverQueryEngine`."

- Hops: 3+
- Path: `(:Class)-[:INHERITS]->(:Class)-[:INHERITS]->...` chain up to root
- Type: structural
- Baseline: fails

### Q13 — Dependency impact
> "Which modules transitively depend on the `VectorStoreIndex` module?"

- Hops: 3
- Path: `(:Module)-[:IMPORTS*1..3]->(:Module containing VectorStoreIndex)`
- Type: structural
- Baseline: fails
- Note: wheel distributions do not contain the upstream test suite

### Q14 — Entry-point anchor
> "Which function is the entry-point anchor for `index.as_query_engine()`?"

- Path: `(:Function {name:"as_query_engine"})`
- Type: structural
- Baseline: partial

### Q15 — Refactor safety
> "Which `NodeWithScore` methods form the class-level refactoring surface?"

- Path: `(:Class {name:"NodeWithScore"})-[:DEFINES]->(:Function)`
- Type: structural
- Baseline: fails

---

## Semantic / Hybrid Queries (Vector + Graph)

### Q16 — Concept + structure
> "Which classes have retry or fallback in their names?"

- Oracle: matching class names
- Type: hybrid
- Baseline: semantic finds some, misses structural relationship

### Q17 — Pattern detection
> "Which modules are named for events, callbacks, hooks, or dispatchers?"

- Oracle: matching module names
- Type: hybrid
- Baseline: partial via semantic

### Q18 — Async naming convention
> "Which async retrieve, query, embed, or generate functions use the conventional a-prefix?"

- Hops: 1 naming lookup
- Type: hybrid
- Baseline: partial

### Q19 — Query pipeline membership
> "Which core query-engine, retriever, and synthesizer classes participate in query flow?"

- Oracle: unordered membership in the core query pipeline
- Type: hybrid (semantic for explanation, graph for path)
- Baseline: partial

### Q20 — Persistence naming
> "Which functions are named for reading, writing, loading, saving, or persistence?"

- Oracle: matching function names
- Type: hybrid
- Baseline: partial

---

## Extended Coverage (Q21-Q50)

| IDs | Cases | Focus |
|-----|------:|-------|
| Q21-Q25 | 5 | Additional direct `DEFINES`, `INHERITS`, `IMPORTS`, and `CALLS` |
| Q26-Q35 | 10 | Composed and bounded two-hop structural traversal |
| Q36-Q40 | 5 | Three-hop call, import, and inheritance traversal |
| Q41-Q45 | 5 | Semantic and hybrid naming-based retrieval |
| Q46-Q50 | 5 | Entity-type ambiguity, callable spelling, and repository scoping |

Together with Q1-Q20, the partitions contain 10 direct structural, 15
two-hop, 10 three-hop, 10 semantic/hybrid, and 5 ambiguity/scoping cases.

Subclass override and inherited-method questions use deterministic composed
plans. The analyzer selects an exact anchor, traverses ordered relationship
steps such as `INHERITS → DEFINES`, optionally filters the terminal method, and
projects the answer-bearing stage instead of returning every intermediate node.

---

## Acceptance Criteria

Acceptance results are valid only with 100% ground-truth coverage and matching
graph/vector provenance. Partial runs are inconclusive and must not be marked
as passing.

| Metric | Target |
|--------|--------|
| Hit Rate @5 (1-hop) | ≥ 0.90 |
| Hit Rate @5 (2-3 hop) | ≥ 0.75 |
| MRR (all) | ≥ 0.70 |
| Faithfulness (RAGAS) | ≥ 0.80 |
| Answer Relevance (RAGAS) | ≥ 0.75 |
| Graph outperforms Vector (2-3 hop) | ΔHitRate ≥ +0.20 |
| Latency (structural queries) | < 1s p95 |
