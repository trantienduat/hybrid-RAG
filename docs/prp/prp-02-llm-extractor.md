# Product Requirement Prompt (PRP) — LLM-assisted Relationship Extractor

## 🎯 Role & Objective
You are an expert AI coder. Your task is to implement the **LLM-assisted Relationship Extractor** module. This module reads file contents and uses a local Large Language Model (via Ollama) to extract implicit semantic relationships—such as class type dependencies (`USES`) on function parameters and return signatures—that statically parsing code via AST tree-sitter cannot easily identify.

---

## 🏛️ Architecture & Tech Stack
*   **Architecture:** Ports and Adapters.
*   **Language:** Python >=3.12.
*   **Libraries:** `httpx` (for calling Ollama REST API), `pydantic` (for parsing JSON structures).

---

## 🛠️ Functional Requirements

### 1. Abstract Port Specification ([ports/llm_extractor.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ports/llm_extractor.py))
Define the abstract port `BaseLLMExtractor` with the following methods:
*   `extract(source_text: str, result: ParseResult) -> list[EdgeData]`: Given file code text and the static AST parse result (which contains known candidate node IDs), return a list of extracted edges.
*   `close()`: Releases client connections.

### 2. Ollama Adapter Implementation ([ingestion/ollama_llm_extractor.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/ollama_llm_extractor.py))
*   Implement `OllamaLLMExtractor` using `httpx.Client`.
*   Maintain connection pooling by instantiating the HTTP client in the constructor.
*   Construct a System Prompt instructing the model to output **valid JSON only** matching the following schema:
    ```json
    {
      "edges": [
        {
          "src_id": " FQN of the caller node (from candidate list) ",
          "rel": "USES",
          "dst_name": " Simple class name (e.g. MyClass) ",
          "confidence": 0.9
        }
      ]
    }
    ```
*   Pass the list of candidate `src_id` values parsed by the AST to prevent the LLM from hallucinating invalid source nodes.
*   Truncate the input code length to fit safely within context parameters.

### 3. Verification & Validation of Outputs
Inside the extractor:
*   Parse the LLM response JSON safely.
*   Filter out standard Python types (e.g., `list`, `dict`, `str`, `int`, `Any`, `Optional`).
*   Verify that `src_id` belongs to the list of known nodes.
*   Resolve `dst_name` to a known node ID (class or function) in the current `ParseResult`. If not found locally, create a stub node.
*   Deduplicate extracted edges against statically resolved AST edges.
*   Filter out edges below a minimum confidence threshold (default: `0.7`).

### 4. Pipeline Integration ([ingestion/pipeline.py](file:///Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG/src/hybrid_rag/ingestion/pipeline.py))
*   Provide a toggle `llm_extract: bool` in `run_indexing_pipeline`. If enabled, call the LLM extractor for each python file and merge the returned edges with the AST output using a merge utility.

---

## 📈 Non-Functional Requirements
*   **Error Tolerance:** If the LLM returns invalid JSON or HTTP errors, log a warning but do not crash the pipeline; fallback to returning an empty list of extracted edges.
*   **Tests:** Implement unit tests mocking the Ollama server responses and validating edge resolution.
