#!/usr/bin/env python3
"""Repeatable local token-footprint and hypothetical hosted-cost comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import tiktoken

EXCLUDE_DIRS = {
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "fixtures",
    "node_modules",
}


def _encoding(model: str) -> Any:
    try:
        encoding_name = tiktoken.encoding_name_for_model(model)
    except KeyError:
        encoding_name = "cl100k_base"
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        raise RuntimeError(
            f"tokenizer {encoding_name!r} is unavailable; install or preload its tiktoken data"
        ) from exc


def scan_local_directory(root: Path, extensions: set[str], model: str) -> tuple[int, int, str]:
    """Return deterministic file/token counts or fail on any unreadable input."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"target is not a local directory: {root}")

    file_count = 0
    errors: list[str] = []
    contents: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        file_count += 1
        contents.append(content)

    if errors:
        raise RuntimeError("unreadable benchmark files: " + "; ".join(errors))
    if file_count == 0:
        raise ValueError("benchmark matched zero readable files")
    encoding = _encoding(model)
    total_tokens = sum(len(encoding.encode(content)) for content in contents)
    if total_tokens == 0:
        raise ValueError("benchmark inputs contain zero tokens")
    return file_count, total_tokens, encoding.name


def calculate_comparison(
    *,
    file_count: int,
    full_context_tokens: int,
    rag_prompt_tokens: int,
    hosted_cost_per_million: float,
    query_count: int,
) -> dict[str, Any]:
    """Calculate an explicitly hypothetical full-context versus RAG scenario."""
    effective_rag_tokens = min(rag_prompt_tokens, full_context_tokens)
    saved_per_query = full_context_tokens - effective_rag_tokens
    full_cost = full_context_tokens * query_count / 1_000_000 * hosted_cost_per_million
    rag_cost = effective_rag_tokens * query_count / 1_000_000 * hosted_cost_per_million
    return {
        "file_count": file_count,
        "full_context_tokens_per_query": full_context_tokens,
        "assumed_rag_prompt_tokens_per_query": rag_prompt_tokens,
        "effective_rag_tokens_per_query": effective_rag_tokens,
        "tokens_avoided_per_query": saved_per_query,
        "token_reduction_percent": round(saved_per_query / full_context_tokens * 100, 4),
        "query_count": query_count,
        "assumed_hosted_input_cost_usd_per_million_tokens": hosted_cost_per_million,
        "hypothetical_full_context_cost_usd": round(full_cost, 6),
        "hypothetical_rag_cost_usd": round(rag_cost, 6),
        "hypothetical_cost_avoided_usd": round(full_cost - rag_cost, 6),
        "evidence_class": "token footprint measurement plus user-supplied scenario assumptions",
        "limitations": (
            "RAG prompt size and hosted price are assumptions; this does not measure actual "
            "Ollama operating cost, generated-answer quality, or a provider invoice."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a local code token footprint and compare explicit cost assumptions."
    )
    parser.add_argument("target", type=Path, help="Local repository directory")
    parser.add_argument("--extensions", default=".py", help="Comma-separated file extensions")
    parser.add_argument("--model", default="gpt-4", help="Tiktoken encoding model")
    parser.add_argument("--cost", type=float, default=3.0, help="Assumed hosted USD/M input")
    parser.add_argument("--rag-size", type=int, default=3000, help="Assumed RAG prompt tokens")
    parser.add_argument("--queries", type=int, default=1000, help="Scenario query count")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.cost < 0:
        parser.error("--cost cannot be negative")
    if args.rag_size < 1:
        parser.error("--rag-size must be at least 1")
    if args.queries < 1:
        parser.error("--queries must be at least 1")
    extensions = {
        value if value.startswith(".") else f".{value}"
        for raw in args.extensions.split(",")
        if (value := raw.strip().lower())
    }
    if not extensions:
        parser.error("--extensions must contain at least one extension")

    try:
        file_count, total_tokens, encoding_name = scan_local_directory(
            args.target, extensions, args.model
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = calculate_comparison(
        file_count=file_count,
        full_context_tokens=total_tokens,
        rag_prompt_tokens=args.rag_size,
        hosted_cost_per_million=args.cost,
        query_count=args.queries,
    )
    result.update(
        {
            "target": str(args.target.resolve()),
            "extensions": sorted(extensions),
            "requested_tokenizer_model": args.model,
            "actual_encoding": encoding_name,
        }
    )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Hybrid-RAG token-footprint scenario")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
