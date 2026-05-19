"""
Context Assembler — format top-N retrieval results into structured LLM context.

Converts a ranked list of retrieval result dicts into a human-readable (and
LLM-readable) context block, with metadata for downstream consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalContext:
    """Assembled retrieval context ready for LLM consumption or display."""

    text: str
    """Formatted text suitable for inclusion in an LLM prompt."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    """Raw result dicts that were included (for inspection / re-ranking)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Aggregate metadata: total_results, shown, sources, has_graph, has_vector."""


class ContextAssembler:
    """Convert ranked retrieval results into a structured context block."""

    def assemble(
        self,
        results: list[dict[str, Any]],
        top_n: int = 5,
        query: str = "",
    ) -> RetrievalContext:
        """
        Format the top *top_n* results into a context block.

        Prioritises results that have text; graph-only entries are included as
        structural annotations so the LLM knows about relevant nodes even when
        their source code was not indexed as a vector chunk.
        """
        selected = results[:top_n]
        lines: list[str] = []

        if query:
            lines.append(f"Query: {query}\n")

        sources_seen: set[str] = set()
        for i, item in enumerate(selected, start=1):
            label = item.get("label", "")
            name = item.get("name", "") or item.get("node_id", "")
            file_path = item.get("file_path", "")
            text = item.get("text", "")
            rel = item.get("rel", "")
            source = item.get("source", "")
            score = item.get("rrf_score") or item.get("score") or 0.0

            sources_seen.add(source)

            header_parts = [f"[{i}]"]
            if label and name:
                header_parts.append(f"{label}: {name}")
            elif name:
                header_parts.append(name)
            if file_path:
                header_parts.append(f"({file_path})")
            if rel:
                header_parts.append(f"via {rel}")
            header_parts.append(f"[score={score:.4f}, src={source}]")

            lines.append(" ".join(header_parts))

            if text:
                lines.append(text.strip())
            else:
                lines.append("[Structural node — no text chunk indexed]")

            lines.append("")  # blank separator

        assembled_text = "\n".join(lines).rstrip()
        return RetrievalContext(
            text=assembled_text,
            chunks=selected,
            metadata={
                "total_results": len(results),
                "shown": len(selected),
                "sources": sorted(s for s in sources_seen if s),
                "has_graph": any(r.get("source") in ("graph", "hybrid") for r in selected),
                "has_vector": any(r.get("source") in ("vector", "hybrid") for r in selected),
            },
        )
