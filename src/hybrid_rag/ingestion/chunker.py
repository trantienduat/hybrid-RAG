"""
AST-aware chunker: extracts actual source code text per KG node.

Produces list[Chunk] from a parsed file — one chunk per Function/Class/Module
node, containing real source code (not just metadata). Large nodes are split
with overlapping sliding windows so nothing exceeds the embedding token budget.

chunk_file()  — main API: parse file → list[Chunk]
chunk_nodes() — lower-level: NodeData list + source lines → list[Chunk]
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_rag.ingestion.parser import NodeData, parse_file

# Rough chars-per-token estimate for nomic-embed-text (cl100k tokenizer ≈4 chars)
_CHARS_PER_TOKEN: int = 4
# Max tokens for nomic-embed-text context window
_DEFAULT_MAX_TOKENS: int = 512
_DEFAULT_OVERLAP_TOKENS: int = 64

# Node labels to produce code chunks for (Module = whole-file header only)
_CHUNK_LABELS: frozenset[str] = frozenset(
    {"Function", "Class", "Module", "Document", "Configuration"}
)


@dataclass
class Chunk:
    """A text chunk ready for embedding and vector upsert."""

    node_id: str
    chunk_index: int  # 0-based; >0 means this node was split
    text: str  # actual source text (possibly with header prefix)
    label: str  # Module | Class | Function
    file_path: str
    start_line: int  # 1-based, line in original file
    end_line: int
    properties: dict[str, Any] = field(default_factory=dict)


# ── Public API ─────────────────────────────────────────────────────────────────


def chunk_file(
    file_path: Path,
    repo_root: Path,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
    repo_name: str = "",
) -> list[Chunk]:
    """Parse file and return source-code chunks for every KG node."""
    result = parse_file(file_path, repo_root, repo_name=repo_name)
    if result.errors and not result.nodes:
        return []

    try:
        source_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    return chunk_nodes(
        result.nodes, source_lines, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )


def chunk_nodes(
    nodes: list[NodeData],
    source_lines: list[str],
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Convert NodeData list + source lines into Chunk list."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
    chunks: list[Chunk] = []

    for node in nodes:
        if node.label not in _CHUNK_LABELS:
            continue

        # External stubs have no source
        if node.label == "Module" and node.properties.get("type") == "external":
            continue

        line_start = node.properties.get("line_start", 1)
        line_end = node.properties.get("line_end", len(source_lines))

        # For Module: use first N lines (imports + module docstring region)
        if node.label == "Module":
            line_end = min(line_end, line_start + _DEFAULT_MAX_TOKENS // 2)

        # Extract source code for this node's span (1-based → 0-based slice)
        node_lines = source_lines[line_start - 1 : line_end]
        node_text = textwrap.dedent("\n".join(node_lines)).strip()

        # Header line for context when split
        header = _make_header(node)

        if len(node_text) <= max_chars:
            chunks.append(
                Chunk(
                    node_id=node.id,
                    chunk_index=0,
                    text=f"{header}\n{node_text}" if header else node_text,
                    label=node.label,
                    file_path=node.properties.get("file_path", ""),
                    start_line=line_start,
                    end_line=line_end,
                    properties={"total_chunks": 1},
                )
            )
        else:
            # Sliding window split
            sub_chunks = _split_text(node_text, max_chars, overlap_chars)
            total = len(sub_chunks)
            for idx, (sub_text, sl, el) in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        node_id=node.id,
                        chunk_index=idx,
                        text=f"{header}\n{sub_text}" if header else sub_text,
                        label=node.label,
                        file_path=node.properties.get("file_path", ""),
                        start_line=line_start + sl,
                        end_line=line_start + el,
                        properties={"total_chunks": total},
                    )
                )

    return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_header(node: NodeData) -> str:
    """One-line context prefix, e.g. '# [Function] MyClass.do_thing (file.py:42)'"""
    p = node.properties
    name = p.get("name", node.id)
    label = node.label
    file_path = p.get("file_path", "")
    line = p.get("line_start", "")

    if label == "Function" and p.get("class_name"):
        display = f"{p['class_name']}.{name}"
    else:
        display = name

    loc = f"{file_path}:{line}" if line else file_path
    return f"# [{label}] {display} ({loc})"


def _split_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """
    Split text into (sub_text, start_line_offset, end_line_offset) tuples.

    Tries to split at newline boundaries to avoid cutting mid-line.
    Returns list of (text_fragment, start_line_offset, end_line_offset).
    """
    lines = text.splitlines()
    result: list[tuple[str, int, int]] = []
    start_line = 0

    while start_line < len(lines):
        # Accumulate lines until budget exhausted
        budget = max_chars
        end_line = start_line
        while end_line < len(lines) and budget > 0:
            budget -= len(lines[end_line]) + 1  # +1 for newline
            if budget >= 0 or end_line == start_line:  # always include at least 1 line
                end_line += 1

        sub_text = "\n".join(lines[start_line:end_line])
        result.append((sub_text, start_line, end_line - 1))

        if end_line >= len(lines):
            break

        # Advance with overlap: step back by overlap_chars worth of lines
        overlap_lines = 0
        overlap_budget = overlap_chars
        for li in range(end_line - 1, start_line, -1):
            overlap_budget -= len(lines[li]) + 1
            if overlap_budget < 0:
                break
            overlap_lines += 1

        next_start = end_line - overlap_lines
        if next_start <= start_line:
            next_start = end_line  # no progress guard
        start_line = next_start

    return result
