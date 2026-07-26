# Evaluation: Multi-hop Test Queries

Target codebase for queries: **LlamaIndex** (Python, high modularity, deep hierarchies)

The Q1-Q20 corpus is a **diagnostic graph benchmark**. Its ground truth is
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
> "Where is the `similarity_top_k` parameter used inside `VectorIndexRetriever`?"

- Hops: 1
- Path: `(:Class {name:"VectorIndexRetriever"})-[:DEFINES]->(:Function)-[:USES]->(:Variable {name:"similarity_top_k"})`
- Type: structural
- Baseline: partial

---

## 2-Hop Queries

### Q6 — Transitive inheritance
> "What methods are available to all subclasses of `BaseIndex` through inheritance?"

- Hops: 2
- Path: `(:Class)-[:INHERITS]->(:Class {name:"BaseIndex"})-[:DEFINES]->(:Function)`
- Type: structural
- Baseline: very weak

### Q7 — Indirect callers
> "Which modules contain functions that call `embed_model.get_text_embedding()`?"

- Hops: 2
- Path: `(:Function)-[:CALLS]->(:Function {name:"get_text_embedding"})<-[:DEFINES]-(:Class)-[:DEFINED_IN]->(:Module)`
- Type: structural
- Baseline: weak

### Q8 — Import chain
> "What does importing `QueryEngine` transitively bring into the namespace?"

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
> "Which modules depend on `StorageContext` and what functions do they use from it?"

- Hops: 2
- Path: `(:Module)-[:IMPORTS]->(:Module {name:"storage_context"})`, then `(:Function)-[:USES]->(:Class {name:"StorageContext"})`
- Type: structural + semantic
- Baseline: partial on semantic, fails on structural

---

## 3-Hop Queries

### Q11 — Impact analysis
> "Which functions can reach `retrieve()` within three calls?"

- Hops: 3
- Path: `(:Function)-[:CALLS*1..3]->(:Function {name:"retrieve"})`
- Type: structural
- Baseline: fails

### Q12 — Deep inheritance chain
> "Trace the full method resolution order (MRO) for `RetrieverQueryEngine`."

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

### Q14 — Cross-cutting concern
> "Trace all code paths from user calling `index.as_query_engine()` to when embeddings are generated."

- Hops: 3+
- Path: `as_query_engine()-[:CALLS]->...-[:CALLS]->embed_model.get_text_embedding()`
- Type: structural
- Baseline: fails

### Q15 — Refactor safety
> "What is the blast radius of renaming the `nodes` parameter in `NodeWithScore`?"

- Hops: 3
- Path: `(:Class {name:"NodeWithScore"})-[:DEFINES]->(:Function)-[:CALLED_BY]->(:Function)-[:DEFINED_IN]->(:Module)`
- Type: structural
- Baseline: fails

---

## Semantic / Hybrid Queries (Vector + Graph)

### Q16 — Concept + structure
> "Find all classes that implement a retry or fallback mechanism and show their inheritance hierarchy."

- Hops: 2 (graph for hierarchy) + semantic search for "retry/fallback"
- Type: hybrid
- Baseline: semantic finds some, misses structural relationship

### Q17 — Pattern detection
> "Which modules implement the observer pattern (event callbacks or hooks)?"

- Hops: 1-2 + semantic
- Type: hybrid
- Baseline: partial via semantic

### Q18 — Docstring + structure
> "Find all functions documented as 'async-safe' and check if they share a common base class."

- Hops: 2 (semantic for docstring, graph for base class)
- Type: hybrid
- Baseline: partial

### Q19 — Architecture query
> "Explain the data flow when a user submits a query: which classes are instantiated and in what order?"

- Hops: N (full call graph from entry point)
- Type: hybrid (semantic for explanation, graph for path)
- Baseline: weak, hallucinates ordering

### Q20 — Privacy/security audit
> "Which functions access the file system directly (open/read/write) and are they reachable from the public API?"

- Hops: 3+ (graph: public API → call chain → fs functions) + semantic: identify fs calls
- Type: hybrid
- Baseline: fails on reachability

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
