"""
Gemini embedder adapter — implements BaseEmbedder port.

Vendor: Google Gemini Cloud API (text-embedding-004, 768-dim or as configured).
Gathers detailed network latency, model latency, and payload token metrics for full LLM Observability.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from hybrid_rag.ingestion.parser import NodeData
from hybrid_rag.ports.embedder import BaseEmbedder
from hybrid_rag.utils.tracing import start_span

logger = logging.getLogger(__name__)

_DEFAULT_GEMINI_MODEL = "text-embedding-004"
_HTTP_TIMEOUT = 30.0
_MAX_TEXT_LENGTH = (
    8000  # Max characters per embedding request to prevent API crashes or truncation errors
)


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


class GeminiEmbedder(BaseEmbedder):
    """BaseEmbedder adapter using Google Gemini's Cloud API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini requests will fail.")
        self._model = model or os.environ.get("EMBED_MODEL", _DEFAULT_GEMINI_MODEL)
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
        Embed multiple text strings in batch requests using Gemini's batchEmbedContents endpoint.
        """
        if not texts:
            return []

        # If API key is missing, return list of zero-vectors
        if not self._api_key:
            return [[0.0] * 768 for _ in texts]

        # Gemini supports batch embedding via batchEmbedContents
        results: list[list[float]] = []
        batch_size = 100  # Gemini limits batch size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._embed_batch(batch))

        return results

    # ── Internal ──────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Call Gemini embedContent REST API endpoint directly."""
        # Truncate text to prevent huge payload crashes
        if len(text) > _MAX_TEXT_LENGTH:
            logger.warning(
                "Embedding text truncated from %d to %d chars", len(text), _MAX_TEXT_LENGTH
            )
            text = text[:_MAX_TEXT_LENGTH]

        if not self._api_key:
            return [0.0] * 768

        retries = 3
        backoff = 1.5

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:embedContent?key={self._api_key}"
        payload = {"model": f"models/{self._model}", "content": {"parts": [{"text": text}]}}

        for attempt in range(1, retries + 1):
            t_start = time.perf_counter()
            with start_span("gemini_embed", {"attempt": attempt, "text_len": len(text)}) as span:
                try:
                    resp = self._client.post(url, json=payload)
                    latency_ms = (time.perf_counter() - t_start) * 1000
                    span.set_attribute("network_latency_ms", latency_ms)

                    resp.raise_for_status()
                    res_json = resp.json()
                    embedding_values = res_json["embedding"]["values"]

                    # Pad or truncate values to target 768-dim if needed
                    # (Google's text-embedding-004 defaults to 768 or can be configured)
                    if len(embedding_values) != 768:
                        if len(embedding_values) > 768:
                            embedding_values = embedding_values[:768]
                        else:
                            embedding_values += [0.0] * (768 - len(embedding_values))

                    return embedding_values

                except Exception as exc:
                    logger.warning(
                        "Gemini embedding attempt %d/%d failed for text (len=%d): %s",
                        attempt,
                        retries,
                        len(text),
                        exc,
                    )
                    span.record_exception(exc)
                    if attempt == retries:
                        logger.error(
                            "All embedding attempts failed: %s. Returning zero-vector.", exc
                        )
                        return [0.0] * 768
                    time.sleep(backoff**attempt)

        return [0.0] * 768

    def _embed_batch(self, batch_texts: list[str]) -> list[list[float]]:
        """Call Gemini batchEmbedContents REST API endpoint directly."""
        if not self._api_key:
            return [[0.0] * 768 for _ in batch_texts]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:batchEmbedContents?key={self._api_key}"

        requests = []
        for text in batch_texts:
            # Truncate text
            if len(text) > _MAX_TEXT_LENGTH:
                text = text[:_MAX_TEXT_LENGTH]
            requests.append(
                {"model": f"models/{self._model}", "content": {"parts": [{"text": text}]}}
            )

        payload = {"requests": requests}
        t_start = time.perf_counter()

        with start_span("gemini_batch_embed", {"batch_size": len(batch_texts)}) as span:
            try:
                resp = self._client.post(url, json=payload)
                latency_ms = (time.perf_counter() - t_start) * 1000
                span.set_attribute("network_latency_ms", latency_ms)

                resp.raise_for_status()
                res_json = resp.json()

                embeddings = []
                for item in res_json.get("embeddings", []):
                    values = item.get("values", [])
                    # Pad or truncate values to target 768-dim
                    if len(values) != 768:
                        if len(values) > 768:
                            values = values[:768]
                        else:
                            values += [0.0] * (768 - len(values))
                    embeddings.append(values)

                # Ensure correct response shape matching batch
                while len(embeddings) < len(batch_texts):
                    embeddings.append([0.0] * 768)
                return embeddings

            except Exception as exc:
                logger.error("Gemini batch embedding failed: %s. Returning zero-vectors.", exc)
                span.record_exception(exc)
                return [[0.0] * 768 for _ in batch_texts]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GeminiEmbedder:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
