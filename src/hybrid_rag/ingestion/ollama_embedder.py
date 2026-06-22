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

from hybrid_rag.constants import DEFAULT_EMBED_MODEL
from hybrid_rag.ingestion.parser import NodeData
from hybrid_rag.ports.embedder import BaseEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = DEFAULT_EMBED_MODEL
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
        self._url = (ollama_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)).rstrip(
            "/"
        )
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
            chunks.append(
                {
                    "node_id": node.id,
                    "label": node.label,
                    "file_path": node.properties.get("file_path", ""),
                    "text": text,
                    "embedding": embedding,
                }
            )
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string for similarity search."""
        return self._embed(query)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple text strings in a batch using Ollama's /api/embed API.
        Falls back to thread-based single embeds if the batch API fails or is not supported.
        """
        if not texts:
            return []

        # 1. Truncate each text to max_safe_len to prevent huge payload crashes (same as single _embed)
        max_safe_len = 8000
        safe_texts = []
        for text in texts:
            if len(text) > max_safe_len:
                logger.warning(
                    "Embedding text truncated from %d to %d chars in batch", len(text), max_safe_len
                )
                safe_texts.append(text[:max_safe_len])
            else:
                safe_texts.append(text)

        # 2. Try the batch embed API
        try:
            resp = self._client.post(
                f"{self._url}/api/embed",
                json={
                    "model": self._model,
                    "input": safe_texts,
                    "keep_alive": os.environ.get("OLLAMA_EMBED_KEEP_ALIVE", "10s"),
                },
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if embeddings and len(embeddings) == len(texts):
                return embeddings
            logger.warning(
                "Batch embedding response format invalid or mismatched length. Falling back to single embeds."
            )
        except Exception as exc:
            logger.warning(
                "Batch embedding via /api/embed failed: %s. Falling back to single/concurrent embeds.",
                exc,
            )

        # 3. Fallback to legacy concurrent/sequential _embed
        import concurrent.futures

        max_workers = int(os.environ.get("EMBED_CONCURRENCY", "8"))
        if max_workers <= 1:
            return [self._embed(t) for t in texts]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._embed, texts))
        return results

    # ── Internal ──────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        # Truncate text to a maximum of 8000 characters to prevent huge payload crashes
        max_safe_len = 8000
        if len(text) > max_safe_len:
            logger.warning("Embedding text truncated from %d to %d chars", len(text), max_safe_len)
            text = text[:max_safe_len]

        import time

        retries = 3
        backoff = 1.5

        for attempt in range(1, retries + 1):
            try:
                resp = self._client.post(
                    f"{self._url}/api/embeddings",
                    json={
                        "model": self._model,
                        "prompt": text,
                        "keep_alive": os.environ.get("OLLAMA_EMBED_KEEP_ALIVE", "10s"),
                    },
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
            except Exception as exc:
                logger.warning(
                    "Ollama embedding attempt %d/%d failed for text (len=%d): %s",
                    attempt,
                    retries,
                    len(text),
                    exc,
                )
                if attempt == retries:
                    # Final attempt fallback: try heavily truncated text
                    try:
                        logger.warning("Retrying with heavily truncated text (500 chars)...")
                        resp = self._client.post(
                            f"{self._url}/api/embeddings",
                            json={
                                "model": self._model,
                                "prompt": text[:500],
                                "keep_alive": os.environ.get("OLLAMA_EMBED_KEEP_ALIVE", "10s"),
                            },
                        )
                        resp.raise_for_status()
                        return resp.json()["embedding"]
                    except Exception as fallback_exc:
                        logger.error(
                            "All embedding attempts failed: %s. Returning zero-vector.",
                            fallback_exc,
                        )
                        return [0.0] * 768
                time.sleep(backoff**attempt)

        return [0.0] * 768

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
