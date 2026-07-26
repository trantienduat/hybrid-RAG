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

QueryType = Literal["local", "global"]
RelationDirection = Literal["in", "out"]

# Global signals: questions about repository-wide architecture, summaries, modules overview
_GLOBAL_RE = re.compile(
    r"\b(?:summarize\s+(?:the\s+)?codebase|codebase\s+summary|repository\s+summary|"
    r"architecture|high-level\s+(?:design|overview)|modules?\s+overview|dependencies\s+flow|"
    r"architectural\s+design|system\s+design|general\s+overview|how\s+is\s+the\s+project\s+structured|"
    r"tóm\s+tắt\s+cấu\s+trúc|tổng\s+quan\s+kiến\s+trúc|sơ\s+đồ\s+hệ\s+thống|"
    r"tổng\s+quan\s+dự\s+án|tóm\s+tắt\s+codebase|luồng\s+hệ\s+thống|cấu\s+trúc\s+thư\s+mục)\b",
    re.IGNORECASE,
)

_RELATION_INTENTS: tuple[tuple[re.Pattern[str], str, RelationDirection, int], ...] = (
    (
        re.compile(r"\b(?:method resolution order|mro)\b", re.IGNORECASE),
        "INHERITS",
        "out",
        3,
    ),
    (
        re.compile(r"\b(?:define[sd]?|methods?\s+(?:does|of|in))\b", re.IGNORECASE),
        "DEFINES",
        "out",
        1,
    ),
    (
        re.compile(r"\b(?:inherit(?:s|ed)?\s+from|subclasses?\s+of)\b", re.IGNORECASE),
        "INHERITS",
        "in",
        1,
    ),
    (
        re.compile(r"\b(?:callers?\s+of|functions?\s+.*\bcall)\b", re.IGNORECASE),
        "CALLS",
        "in",
        1,
    ),
    (re.compile(r"\b(?:calls?|invokes?)\b", re.IGNORECASE), "CALLS", "out", 1),
    (re.compile(r"\bimport(?:s|ing|ed)?\b", re.IGNORECASE), "IMPORTS", "out", 1),
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
        # Vietnamese stop words
        "về",
        "của",
        "và",
        "hoặc",
        "là",
        "cho",
        "từ",
        "đến",
        "trong",
        "trên",
        "dưới",
        "đi",
        "này",
        "đó",
        "kia",
        "nào",
        "gì",
        "sao",
        "thế",
        "như",
        "được",
        "bị",
        "bởi",
        "các",
        "những",
        "một",
        "hai",
        "ba",
        "ra",
        "vào",
        "lại",
        "qua",
        "theo",
        "với",
        "tại",
        "cùng",
        "ở",
        "mỗi",
        "từng",
        "tự",
        "chỉ",
        "cả",
        "mô",
        "tả",
        "giải",
        "thích",
    }
)


@dataclass
class QueryAnalysis:
    """Result of query classification and entity extraction."""

    query_type: QueryType
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    relation: str | None = None
    direction: RelationDirection | None = None
    max_hops: int = 1

    def __repr__(self) -> str:
        return (
            f"QueryAnalysis(type={self.query_type!r}, "
            f"entities={self.entities!r}, keywords={self.keywords!r}, "
            f"relation={self.relation!r}, direction={self.direction!r}, "
            f"max_hops={self.max_hops!r})"
        )


def analyze(query: str) -> QueryAnalysis:
    """Classify a free-text query and extract code identifiers."""
    is_global = bool(_GLOBAL_RE.search(query))
    query_type: QueryType = "global" if is_global else "local"
    relation = None
    direction = None
    max_hops = 1
    for pattern, candidate_relation, candidate_direction, candidate_hops in _RELATION_INTENTS:
        if pattern.search(query):
            relation = candidate_relation
            direction = candidate_direction
            max_hops = candidate_hops
            break
    if relation and re.search(r"\btransitiv(?:e|ely)\b", query, re.IGNORECASE):
        max_hops = 3

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
    for m in re.finditer(r"\b(\w{2,})\b", query):
        w = m.group(1).lower()
        # Ensure it contains at least one letter (avoid pure numbers/underscores as keywords)
        if not any(c.isalpha() for c in w):
            continue
        if w not in _STOP_WORDS and w not in entity_lower and w not in kw_seen:
            kw_seen.add(w)
            keywords.append(w)

    return QueryAnalysis(
        query_type=query_type,
        entities=list(dict.fromkeys(entities)),
        keywords=list(dict.fromkeys(keywords)),
        relation=relation,
        direction=direction,
        max_hops=max_hops,
    )
