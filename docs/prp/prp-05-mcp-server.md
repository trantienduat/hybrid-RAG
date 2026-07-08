# Product Requirement Prompt (PRP) — Model Context Protocol (MCP) Server

## 🎯 Role & Objective
You are an expert AI protocol engineer. Your task is to implement the **Model Context Protocol (MCP) Server** adapter for the `hybrid-RAG` platform. The server must expose the platform's core retrieval, graph query, and community partitioning capabilities as standardized tools and resources, allowing external LLM clients (like Cursor, Cline, or Claude Desktop) to invoke them over `stdio` or `SSE` transport channels.

---

## 🏛️ Tech Stack & Protocols
*   **Protocol:** Model Context Protocol (MCP) by Anthropic.
*   **Language:** Python >=3.12.
*   **Libraries:** `mcp` SDK, `fastmcp` (for simplified tool/resource registration), `typer` (CLI bindings).

---

## 🛠️ Functional Requirements

### 1. FastMCP Instance & Configuration
Initialize a FastMCP server named "Hybrid-RAG" loading configuration parameters dynamically from environment variables (`FALKORDB_HOST`, `QDRANT_HOST`).

### 2. Exposed Tools
Expose the following Python functions decorated as MCP tools:
*   `query_codebase(question: str, repository: str = None, top_k: int = 20) -> str`: Runs the `HybridRetriever` retrieval engine and returns the assembled Markdown context snippet.
*   `list_repositories() -> list[str]`: Queries FalkorDB for unique repository namespaces.
*   `search_ast_nodes(query: str, repository: str = None) -> str`: Performs text lookup for classes or functions matching the prefix query.
*   `get_ast_neighbors(node_id: str) -> str`: Returns direct structural links (callers, overrides) for the target node ID.
*   `get_community_report(repository: str) -> str`: Fetches Directory-based community report partition summaries.

### 3. Exposed Resources & Prompts
*   Expose the graph schema as a resource: `schema://graph` (yielding node types and Cypher relation schema).
*   Provide a pre-packaged prompt template: `debug-bug-trace` instructing client LLMs on how to query node paths to resolve exception stack traces.

### 4. Transport Execution Options
Modify the CLI `mcp` command to support:
*   **stdio transport:** Listens on standard input and output streams. Required for Claude Desktop and Cline local configurations.
*   **SSE transport:** Starts a web server on a specified host and port (default: `8001`) yielding server-sent event frames.

---

## 📈 Non-Functional Requirements & Testing
*   **JSON-RPC Conformity:** Ensure logs do not print to `stdout` in `stdio` mode to prevent corrupting JSON-RPC communication frames (use `stderr` for logging).
*   **Unit Tests:** Implement unit tests verifying the MCP server registers all five tools and runs properly on mock inputs.
