"""
Context Assembler — format top-N retrieval results into structured LLM context.

Converts a ranked list of retrieval result dicts into a human-readable (and
LLM-readable) context block, with metadata for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from hybrid_rag.config import app_config, translate_path_for_docker
from hybrid_rag.retrieval.skeletonizer import skeletonize_file


@dataclass
class RetrievalContext:
    """Assembled retrieval context ready for LLM consumption or display."""

    text: str
    """Formatted text suitable for inclusion in an LLM prompt."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    """Raw result dicts that were included (for inspection / re-ranking)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Aggregate metadata: total_results, shown, sources, has_graph, has_vector."""

    timings: dict[str, float] = field(default_factory=dict)
    """Query and retrieval performance breakdown timings in milliseconds."""


class ContextAssembler:
    """Convert ranked retrieval results into a structured context block."""

    def assemble(
        self,
        results: list[dict[str, Any]],
        top_n: int = 5,
        max_tokens: int | None = None,
        max_chars: int | None = None,
        token_estimator: callable | None = None,
        query: str = "",
    ) -> RetrievalContext:
        """
        Format retrieval results into a structured context block.

        If either max_tokens or max_chars is provided, dynamic budgeting is applied:
        chunks are packed strictly in rank order until the budget limit is reached.
        Otherwise, fallback to hardcoded top_n results.
        """
        # Set up token estimator
        if token_estimator is None:
            # Safe offline heuristic: ~1 token ≈ 4 characters of code/text.
            def token_estimator(text):
                return len(text) // 4

        use_budget = (max_tokens is not None) or (max_chars is not None)

        lines: list[str] = []
        if query:
            lines.append(f"Query: {query}\n")

        query_overhead_len = len("\n".join(lines))
        query_overhead_tokens = token_estimator("\n".join(lines)) if lines else 0

        current_tokens = query_overhead_tokens
        current_chars = query_overhead_len

        # Pre-group focus names by file path to support skeletonization consolidation
        file_focus_names: dict[str, list[str]] = {}
        for item in results:
            fp = item.get("file_path", "")
            name = item.get("name", "")
            if fp and name:
                if fp not in file_focus_names:
                    file_focus_names[fp] = []
                file_focus_names[fp].append(name)

        processed_files: set[str] = set()
        chunks_included: list[dict[str, Any]] = []
        chunks_excluded: list[dict[str, Any]] = []
        sources_seen: set[str] = set()

        for i, item in enumerate(results, start=1):
            file_path = item.get("file_path", "")
            repo_name = item.get("repository", "")

            # Check if we should skeletonize this file
            skeletonized_text = None
            if file_path and repo_name:
                if file_path in processed_files:
                    # Skip duplicate class/file blocks to save space and avoid redundancy
                    continue
                processed_files.add(file_path)

                config_path = app_config.get_repo_path(repo_name)
                if config_path:
                    effective_repo_path = translate_path_for_docker(config_path)
                    abs_file_path = Path(effective_repo_path) / file_path
                    focus_names = file_focus_names.get(file_path, [])
                    ext = abs_file_path.suffix.lower()
                    lang = "python" if ext == ".py" else ("java" if ext == ".java" else "python")
                    skeletonized_text = skeletonize_file(abs_file_path, focus_names, language=lang)

            label = item.get("label", "")
            name = item.get("name", "") or item.get("node_id", "")
            text = skeletonized_text if skeletonized_text else item.get("text", "")
            rel = item.get("rel", "")
            source = item.get("source", "")
            score = item.get("rrf_score") or item.get("score") or 0.0

            # Pre-format this individual chunk block
            header_parts = [f"[{i}]"]
            if label and name:
                header_parts.append(f"{label}: {name}")
            elif name:
                header_parts.append(name)
            if file_path:
                header_parts.append(f"({file_path})")
            if skeletonized_text:
                header_parts.append("[Skeletonized]")
            if rel:
                header_parts.append(f"via {rel}")
            header_parts.append(f"[score={score:.4f}, src={source}]")

            chunk_lines = [" ".join(header_parts)]
            if text:
                chunk_lines.append(text.strip())
            else:
                chunk_lines.append("[Structural node — no text chunk indexed]")
            chunk_lines.append("")  # blank separator

            chunk_block = "\n".join(chunk_lines)
            chunk_tokens = token_estimator(chunk_block)
            chunk_chars = len(chunk_block)

            # Evaluate budget constraint
            if use_budget:
                violated = False
                if max_tokens is not None and (current_tokens + chunk_tokens) > max_tokens:
                    violated = True
                if max_chars is not None and (current_chars + chunk_chars) > max_chars:
                    violated = True

                if violated:
                    # Once a high-ranked chunk violates the budget, exclude it and all remaining
                    # chunks to strictly preserve rank priority without packing holes or truncating.
                    chunks_excluded.extend(results[i - 1 :])
                    break

            # Fallback legacy constraint
            elif len(chunks_included) >= top_n:
                chunks_excluded.extend(results[i - 1 :])
                break

            # Pack chunk
            current_tokens += chunk_tokens
            current_chars += chunk_chars
            lines.extend(chunk_lines[:-1])  # add all lines except the trailing blank separator line
            lines.append("")  # explicit separator
            chunks_included.append(item)
            if source:
                sources_seen.add(source)

        assembled_text = "\n".join(lines).rstrip()
        actual_tokens = token_estimator(assembled_text)

        budget_limit = (
            max_tokens if max_tokens is not None else (max_chars if max_chars is not None else 0)
        )

        return RetrievalContext(
            text=assembled_text,
            chunks=chunks_included,
            metadata={
                "total_results": len(results),
                "shown": len(chunks_included),
                "sources": sorted(s for s in sources_seen if s),
                "has_graph": any(r.get("source") in ("graph", "hybrid") for r in chunks_included),
                "has_vector": any(r.get("source") in ("vector", "hybrid") for r in chunks_included),
                "use_budget": use_budget,
                "total_tokens": actual_tokens,
                "total_chars": len(assembled_text),
                "budget_limit": budget_limit,
                "chunks_included_count": len(chunks_included),
                "chunks_excluded_count": len(chunks_excluded),
            },
        )
