"""
Ollama LLM extractor adapter — implements BaseLLMExtractor port.

Vendor: Ollama local inference server (qwen2.5-coder:7b or any code-capable model).

Extracts USES relationships from type annotations that AST cannot capture:
  - Typed function parameters: def foo(self, x: SomeClass)
  - Return type annotations:   def foo(self) -> SomeClass
  - Instance variable annotations: self.engine: QueryEngine = ...

The LLM is prompted to output structured JSON; confidence-filtered results
are converted to EdgeData and returned. All errors are swallowed (returns []).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from hybrid_rag.ingestion.parser import EdgeData, NodeData, ParseResult
from hybrid_rag.ports.llm_extractor import BaseLLMExtractor

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5-coder:7b"
_HTTP_TIMEOUT = 60.0
_MIN_CONFIDENCE = 0.7

# Python builtins and typing primitives the LLM should NOT emit as dst_name
_BUILTIN_TYPES: frozenset[str] = frozenset({
    "str", "int", "float", "bool", "bytes", "bytearray", "memoryview",
    "list", "dict", "set", "frozenset", "tuple", "range", "slice",
    "type", "object", "None", "NoneType", "Ellipsis",
    "Optional", "Union", "Any", "List", "Dict", "Set", "Tuple",
    "Callable", "Type", "ClassVar", "Final", "Literal", "TypeVar",
    "Sequence", "Iterable", "Iterator", "Generator", "AsyncGenerator",
    "Awaitable", "Coroutine", "AsyncIterator", "AsyncIterable",
    "IO", "TextIO", "BinaryIO", "Pattern", "Match",
    "Path", "PurePath", "PosixPath", "WindowsPath",
    "datetime", "date", "time", "timedelta",
    "UUID", "Decimal", "Enum", "IntEnum", "Flag",
    "T", "K", "V",  # common TypeVar names
    "Self", "Never", "NoReturn", "LiteralString",
    "override", "dataclass", "abstractmethod", "property",
    "staticmethod", "classmethod",
    "Exception", "BaseException", "ValueError", "TypeError",
    "RuntimeError", "NotImplementedError", "AttributeError",
    "KeyError", "IndexError", "StopIteration", "OSError",
    "IOError", "FileNotFoundError", "PermissionError",
})

_SYSTEM_PROMPT = (
    "You are a code analysis assistant that extracts semantic relationships "
    "from Python source code. Your output must be valid JSON only — no explanations, "
    "no markdown, no text before or after the JSON."
)

_USER_TEMPLATE = """\
Extract USES relationships from type annotations in the following Python code.
A USES relationship means: a function/method uses a user-defined class as a type annotation
on a parameter, return type, or instance variable assignment.

Rules:
- "src_id" must be EXACTLY one of the node IDs from "Known Nodes" list below
- "dst_name" is the simple class name (e.g. "BaseRetriever", not "pkg.BaseRetriever")
- Only include user-defined classes — skip all Python builtins and common typing primitives
  (str, int, bool, list, dict, Optional, Any, Union, Callable, Path, datetime, etc.)
- "confidence" must be >= 0.7 to include
- If nothing to extract, return {{"edges": []}}

Known Nodes (src_id candidates):
{node_list}

Source code:
{source_code}

Return JSON object only:
{{"edges": [{{"src_id": "...", "rel": "USES", "dst_name": "...", "confidence": 0.9}}, ...]}}
"""


class OllamaLLMExtractor(BaseLLMExtractor):
    """BaseLLMExtractor adapter using Ollama's HTTP generation API."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: float = _HTTP_TIMEOUT,
    ) -> None:
        self._url = (ollama_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)).rstrip("/")
        self._model = model or os.environ.get("LLM_MODEL", _DEFAULT_MODEL)
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ── BaseLLMExtractor interface ────────────────────────────────

    def extract(self, source_text: str, result: ParseResult) -> list[EdgeData]:
        """
        Extract USES edges from type annotations in source_text.

        Uses the node IDs in result as candidate src_ids.
        Returns [] on any error (never raises).
        """
        # Only process nodes with real file paths (skip external stubs)
        local_nodes = [
            n for n in result.nodes
            if n.properties.get("file_path") and n.properties.get("type") != "external"
        ]
        if not local_nodes:
            return []

        prompt = self._build_prompt(source_text, local_nodes)
        raw_response = self._call_ollama(prompt)
        if raw_response is None:
            return []

        return self._parse_response(raw_response, result)

    def close(self) -> None:
        self._client.close()

    # ── Internal helpers ──────────────────────────────────────────

    def _build_prompt(self, source_text: str, nodes: list[NodeData]) -> str:
        node_list = "\n".join(
            f"  {n.id}  ({n.label}: {n.properties.get('name', '')})"
            for n in nodes
        )
        # Truncate very long source to avoid exceeding context window (keep ~6000 chars)
        code = source_text[:6000]
        if len(source_text) > 6000:
            code += "\n# ... (truncated)"
        return _USER_TEMPLATE.format(node_list=node_list, source_code=code)

    def _call_ollama(self, prompt: str) -> str | None:
        """POST to Ollama /api/generate. Returns response text or None on failure."""
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{prompt}",
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 1024,
                "top_p": 1.0,
            },
        }
        try:
            resp = self._client.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except httpx.HTTPError as exc:
            logger.warning("LLM extractor HTTP error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM extractor unexpected error: %s", exc)
            return None

    def _parse_response(self, raw: str, result: ParseResult) -> list[EdgeData]:
        """Parse JSON response and convert to EdgeData list."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("LLM extractor JSON parse failed: %s — raw: %.200s", exc, raw)
            return []

        edges_data = data if isinstance(data, list) else data.get("edges", [])
        if not isinstance(edges_data, list):
            logger.warning("LLM extractor unexpected JSON shape: %s", type(data))
            return []

        # Build lookup sets for validation
        valid_src_ids = {n.id for n in result.nodes}
        existing_edges = {(e.src_id, e.rel, e.dst_id) for e in result.edges}

        results: list[EdgeData] = []
        for item in edges_data:
            if not isinstance(item, dict):
                continue

            src_id = item.get("src_id", "")
            rel = item.get("rel", "USES")
            dst_name = item.get("dst_name", "")
            confidence = float(item.get("confidence", 1.0))

            # Validation
            if not src_id or not dst_name:
                continue
            if src_id not in valid_src_ids:
                logger.debug("LLM extractor: unknown src_id %r — skipping", src_id)
                continue
            if confidence < _MIN_CONFIDENCE:
                continue
            if dst_name in _BUILTIN_TYPES:
                continue
            # Strip generic parameters e.g. "List[Node]" → "Node" won't help, but
            # catch simple cases like "Optional[BaseRetriever]" → skip (complex type)
            if "[" in dst_name or "|" in dst_name:
                continue
            if rel not in ("USES", "CALLS", "INHERITS"):
                continue

            # Resolve dst_name → dst_id
            dst_id = self._resolve_dst(dst_name, result)

            key = (src_id, rel, dst_id)
            if key in existing_edges:
                continue

            results.append(EdgeData(
                src_id=src_id,
                rel=rel,
                dst_id=dst_id,
                properties={"source": "llm", "confidence": confidence, "dst_name": dst_name},
            ))
            existing_edges.add(key)

        logger.debug("LLM extractor: %d new edges extracted", len(results))
        return results

    def _resolve_dst(self, dst_name: str, result: ParseResult) -> str:
        """
        Resolve a simple class name to a known node ID, or return the name as a stub ID.

        Prefers Class nodes, then Function nodes. If not found, returns dst_name directly
        so the entity resolver can handle cross-file resolution later.
        """
        # Exact match first
        for node in result.nodes:
            if node.id == dst_name:
                return node.id

        # Match by simple name for Class nodes (most common case for USES)
        for node in result.nodes:
            if node.label == "Class" and node.properties.get("name") == dst_name:
                return node.id

        # Match by simple name for Function nodes (CALLS case)
        for node in result.nodes:
            if node.label == "Function" and node.properties.get("name") == dst_name:
                return node.id

        # Not found locally — return as-is; entity_resolver will handle cross-file
        return dst_name
