"""
Query Analyzer — classify retrieval queries and extract code identifiers.

Classifies queries as:
  structural — asks about code structure (inheritance, imports, call graph, …)
  semantic   — asks about meaning/behavior (explain, describe, how does X work)
  hybrid     — both signals present, or neither (default)

Extracts:
  entities   — code identifiers (PascalCase, snake_case, backtick/quote delimited)
  keywords   — meaningful non-stopword terms for semantic search
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

QueryType = Literal["structural", "semantic", "hybrid", "global"]

# Global signals: questions about repository-wide architecture, summaries, modules overview
_GLOBAL_RE = re.compile(
    r"\b(?:summarize\s+(?:the\s+)?codebase|codebase\s+summary|repository\s+summary|"
    r"architecture|high-level\s+(?:design|overview)|modules?\s+overview|dependencies\s+flow|"
    r"architectural\s+design|system\s+design|general\s+overview|how\s+is\s+the\s+project\s+structured|"
    r"tóm\s+tắt\s+cấu\s+trúc|tổng\s+quan\s+kiến\s+trúc|sơ\s+đồ\s+hệ\s+thống)\b",
    re.IGNORECASE,
)

# Structural signals: questions about code topology and relationships
_STRUCTURAL_RE = re.compile(
    r"\b(?:inherits?|extends?|implements?|subclass(?:es)?|overrides?|"
    r"imports?|depends?|dependency|dependencies|"
    r"calls?|invokes?|"
    r"defines?|declares?|"
    r"relations?|connections?|graph|paths?|reaches?|"
    r"ancestor|descendant|parent|child(?:ren)?|"
    r"which\s+(?:classes?|functions?|methods?|modules?))\b",
    re.IGNORECASE,
)

# Semantic signals: questions about meaning, intent, or behavior
_SEMANTIC_RE = re.compile(
    r"\b(?:explain|describes?|summarize[sd]?|"
    r"what\s+does|how\s+does|why\s+does|how\s+(?:it\s+)?works?|"
    r"purpose|meaning|intent|example|"
    r"behavior|logic|algorithm|pattern|workflow|handles?|processes?)\b",
    re.IGNORECASE,
)

# PascalCase with ≥2 uppercase letters: BaseEmbedder, GraphStore, FalkorDB
_PASCAL_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)\b")
# snake_case with ≥1 underscore: parse_repo, embed_query, ollama_llm_extractor
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
# Backtick or double-quote delimited identifiers
_QUOTED_RE = re.compile(r"`([^`]+)`|\"([A-Za-z_][A-Za-z0-9_]*)\"")

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "used",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "of",
        "or",
        "and",
        "but",
        "if",
        "as",
        "with",
        "this",
        "that",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "it",
        "its",
        "my",
        "your",
        "our",
        "their",
        "all",
        "any",
        "each",
        "every",
        "some",
        "no",
        "not",
        "show",
        "list",
        "get",
        "find",
        "tell",
        "give",
        "print",
        "code",
        "file",
        "class",
        "function",
        "method",
        "module",
        "between",
        "about",
        "also",
        "more",
        "than",
        "just",
        "only",
    }
)


@dataclass
class QueryAnalysis:
    """Result of query classification and entity extraction."""

    query_type: QueryType
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"QueryAnalysis(type={self.query_type!r}, "
            f"entities={self.entities!r}, keywords={self.keywords!r})"
        )


def analyze(query: str) -> QueryAnalysis:
    """Classify a free-text query and extract code identifiers."""
    is_global = bool(_GLOBAL_RE.search(query))
    if is_global:
        query_type: QueryType = "global"
    else:
        structural = bool(_STRUCTURAL_RE.search(query))
        semantic = bool(_SEMANTIC_RE.search(query))

        if structural and semantic:
            query_type = "hybrid"
        elif structural:
            query_type = "structural"
        elif semantic:
            query_type = "semantic"
        else:
            query_type = "hybrid"  # default: use both graph + vector

    # ── entity extraction ──────────────────────────────────────────
    seen: set[str] = set()
    entities: list[str] = []

    def _add(name: str) -> None:
        stripped = name.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            entities.append(stripped)

    for m in _QUOTED_RE.finditer(query):
        _add(m.group(1) or m.group(2) or "")
    for m in _PASCAL_RE.finditer(query):
        _add(m.group(1))
    for m in _SNAKE_RE.finditer(query):
        _add(m.group(1))

    # ── keyword extraction ─────────────────────────────────────────
    keywords: list[str] = []
    kw_seen: set[str] = set()
    entity_lower = {e.lower() for e in entities}
    for m in re.finditer(r"\b([a-zA-Z][a-zA-Z0-9]{2,})\b", query):
        w = m.group(1).lower()
        if w not in _STOP_WORDS and w not in entity_lower and w not in kw_seen:
            kw_seen.add(w)
            keywords.append(w)

    return QueryAnalysis(
        query_type=query_type,
        entities=list(dict.fromkeys(entities)),
        keywords=list(dict.fromkeys(keywords)),
    )
