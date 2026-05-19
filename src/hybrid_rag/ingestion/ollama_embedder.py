"""
Ollama embedder adapter — implements BaseEmbedder port.

Vendor: Ollama local inference server (nomic-embed-text, 768-dim).
Swap this file for a different adapter (e.g. openai_embedder.py) to change backends.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from hybrid_rag.ingestion.parser import NodeData
from hybrid_rag.ports.embedder import BaseEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_HTTP_TIMEOUT = 30.0


def _node_to_text(node: NodeData) -> str:
    """Render a NodeData into embeddable text."""
    p = node.properties
    parts: list[str] = [f"[{node.label}] {p.get('name', node.id)}"]

    if node.label == "Module":
        parts.append(f"file: {p.get('file_path', '')}")
        if p.get("type") and p["type"] != "external":
            parts.append(f"type: {p['type']}")

    elif node.label == "Class":
        parts.append(f"file: {p.get('file_path', '')}")
        if p.get("docstring"):
            parts.append(p["docstring"])
        if p.get("is_abstract"):
            parts.append("abstract class")

    elif node.label == "Function":
        sig = p.get("signature", "")
        if sig:
            parts.append(sig)
        if p.get("class_name"):
            parts.append(f"method of: {p['class_name']}")
        if p.get("docstring"):
            parts.append(p["docstring"])
        flags = []
        if p.get("is_async"):
            flags.append("async")
        if p.get("is_property"):
            flags.append("property")
        if flags:
            parts.append(" ".join(flags))

    elif node.label == "Variable":
        parts.append(f"file: {p.get('file_path', '')}")
        if p.get("type_annotation"):
            parts.append(f"type: {p['type_annotation']}")

    return "\n".join(parts)


class OllamaEmbedder(BaseEmbedder):
    """BaseEmbedder adapter using Ollama's HTTP embedding API."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._url = (ollama_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)).rstrip("/")
        self._model = model or os.environ.get("EMBED_MODEL", _DEFAULT_MODEL)
        self._client = httpx.Client(timeout=_HTTP_TIMEOUT)

    # ── BaseEmbedder interface ────────────────────────────────────

    def embed_nodes(self, nodes: list[NodeData]) -> list[dict[str, Any]]:
        """
        Embed each node. Returns list of chunk dicts ready for VectorStore.upsert().

        Skips external stub modules (no useful text to embed).
        """
        chunks: list[dict[str, Any]] = []
        for node in nodes:
            if node.label == "Module" and node.properties.get("type") == "external":
                continue
            text = _node_to_text(node)
            embedding = self._embed(text)
            chunks.append({
                "node_id": node.id,
                "label": node.label,
                "file_path": node.properties.get("file_path", ""),
                "text": text,
                "embedding": embedding,
            })
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string for similarity search."""
        return self._embed(query)

    # ── Internal ──────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        resp = self._client.post(
            f"{self._url}/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaEmbedder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
