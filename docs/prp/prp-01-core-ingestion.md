# Product Requirement Prompt (PRP) — Core Ingestion Pipeline & AST Parsing

## 🎯 Role & Objective
You are an expert software engineer AI. Your task is to implement the **Core Ingestion Pipeline and AST Parsing** module for the `hybrid-RAG` codebase understanding platform. This module must parse a local Python repository, build a syntactic relationship graph, split code into structural chunks, and populate both FalkorDB (graph) and Qdrant (vector) databases.

---

## 🏛️ Architecture & Tech Stack
*   **Architecture:** Ports and Adapters (Hexagonal). Core logic must interact with databases only via abstract port interfaces.
*   **Language:** Python >=3.12.
*   **Libraries:** `tree-sitter`, `tree-sitter-python`, `falkordb`, `qdrant-client`, `httpx`, `typer` (CLI).

---

## 🛠️ Functional Requirements

### 1. Abstract Ports ([ports/](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ports/))
Define abstract base classes for the database clients:
*   `BaseGraphStore`: Methods for `ingest(parse_result)`, `find_nodes(query)`, and `find_neighbors(node_id)`.
*   `BaseVectorStore`: Methods for `upsert(chunks)` and `search(vector, limit)`.
*   `BaseEmbedder`: Methods for `embed_texts(texts)` and `embed_query(query)`.

### 2. AST Parser ([ingestion/parser.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/parser.py))
*   Parse local Python files recursively using `tree-sitter-python`.
*   Extract code nodes with labels: `Module`, `Class`, `Function`, `Variable`.
*   Record node properties: `id` (Fully Qualified Name - FQN, e.g. `path/to/file.py::ClassName::method`), `name`, `file_path`, `start_line`, `end_line`, `docstring`, `signature`, and flags (`is_async`, `is_abstract`).
*   Identify static relationships:
    *   `DEFINES`: Nesting hierarchy (Module defines Class, Class defines Function).
    *   `INHERITS`: Base class inheritance links.
    *   `CALLS`: Simple call expressions inside functions to other local functions.

### 3. Entity Resolver ([ingestion/entity_resolver.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/entity_resolver.py))
*   Deduplicate import declarations.
*   Merge duplicate local definitions and resolve external/standard library stubs, compiling a unified, clean graph result representation.

### 4. AST-Aware Chunker ([ingestion/chunker.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/chunker.py))
*   Implement a chunker that splits source files along function and class boundaries (based on start/end lines from AST nodes) rather than raw character splits, preserving context.

### 5. Ingestion Influx ([graph/falkordb_store.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/graph/falkordb_store.py) & [vector/qdrant_store.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/vector/qdrant_store.py))
*   **FalkorDB Store:** Build Cypher queries to merge nodes and relationships. Protect against query injections and format parameters correctly.
*   **Qdrant Store:** Manage vector collections, configure distance metrics (Cosine), and upsert code chunk vectors.

### 6. CLI Entrypoint ([cli.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/cli.py))
*   Expose a `hybrid-rag index <repo_path>` Typer CLI command.

---

## 📈 Non-Functional Requirements & Safety
*   **Isolation:** All database stores must catch connection errors and raise informative custom exceptions.
*   **Validation:** Use `pydantic` schemas for API data boundaries.
*   **Tests:** Implement unit tests verifying tree-sitter node extraction and FalkorDB Cypher generation using mocks.
