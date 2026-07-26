"""
Query Analyzer — route retrieval queries and extract code identifiers.

Routes queries as:
  local  — entity, relationship, or behavior lookup (default)
  global — repository-wide architecture or summary request

Extracts:
  entities   — code identifiers (PascalCase, snake_case, backtick/quote delimited)
  keywords   — meaningful non-stopword terms for semantic search
  graph_plan — deterministic ordered traversal for supported composed intents
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

QueryType = Literal["local", "global"]
RelationDirection = Literal["in", "out"]
TargetLabel = Literal["Class", "Function", "Module"]


@dataclass(frozen=True)
class RelationStep:
    """One directed relationship traversal in a deterministic graph plan."""

    relation: str
    direction: RelationDirection
    max_hops: int = 1


@dataclass(frozen=True)
class GraphPlan:
    """Ordered graph traversal with an explicit answer projection."""

    anchor: str
    steps: tuple[RelationStep, ...]
    result_step: int
    terminal_name: str | None = None


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
        re.compile(r"\b(?:ancestors?|ancestor\s+hierarchy)\b", re.IGNORECASE),
        "INHERITS",
        "out",
        1,
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
        re.compile(
            r"\bfunctions?\s+(?:are\s+)?(?:reachable\s+from|called\s+by)\b",
            re.IGNORECASE,
        ),
        "CALLS",
        "out",
        1,
    ),
    (
        re.compile(
            r"\b(?:functions?|methods?)\b.*\b(?:reach|lead\s+to)\b",
            re.IGNORECASE,
        ),
        "CALLS",
        "in",
        1,
    ),
    (
        re.compile(r"\b(?:depend(?:s|ed|ing)?\s+on|dependenc(?:y|ies)\s+of)\b", re.IGNORECASE),
        "IMPORTS",
        "in",
        1,
    ),
    (
        re.compile(
            r"\b(?:callers?\s+of|functions?\s+.*\b(?:call|invoke))",
            re.IGNORECASE,
        ),
        "CALLS",
        "in",
        1,
    ),
    (
        re.compile(
            r"\b(?:what|which)\b.*\b(?:does|do)\b.*\b(?:call|invoke)s?\b",
            re.IGNORECASE,
        ),
        "CALLS",
        "out",
        1,
    ),
    (re.compile(r"\bimport(?:s|ing|ed)?\b", re.IGNORECASE), "IMPORTS", "out", 1),
)

# PascalCase with ≥2 uppercase letters: BaseEmbedder, GraphStore, FalkorDB
_PASCAL_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)\b")
# snake_case with ≥1 underscore: parse_repo, embed_query, ollama_llm_extractor
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
# Backtick or double-quote delimited identifiers
_QUOTED_RE = re.compile(r"`([^`]+)`|\"([A-Za-z_][A-Za-z0-9_]*)\"")
# Function-call syntax: retrieve(), as_query_engine(), BaseRetriever.retrieve()
_CALLABLE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\)")
_HOP_COUNT_RE = re.compile(
    r"\b(?:within|up\s+to|at\s+most)\s+"
    r"(?P<count>\d+|one|two|three|four|five)\s+"
    r"(?:(?:call|import|inheritance)\s+)?(?:calls?|hops?|levels?|steps?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_TARGET_LABEL_RE = re.compile(
    r"\b(?:which|what|find|list|show)\b.{0,100}?"
    r"\b(?P<kind>classes?|functions?|methods?|modules?)\b",
    re.IGNORECASE,
)
_TARGET_LABELS: dict[str, TargetLabel] = {
    "class": "Class",
    "classes": "Class",
    "function": "Function",
    "functions": "Function",
    "method": "Function",
    "methods": "Function",
    "module": "Module",
    "modules": "Module",
}

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
    graph_plan: GraphPlan | None = None
    target_label: TargetLabel | None = None

    def __repr__(self) -> str:
        return (
            f"QueryAnalysis(type={self.query_type!r}, "
            f"entities={self.entities!r}, keywords={self.keywords!r}, "
            f"relation={self.relation!r}, direction={self.direction!r}, "
            f"max_hops={self.max_hops!r}, graph_plan={self.graph_plan!r}, "
            f"target_label={self.target_label!r})"
        )


def _build_graph_plan(query: str, entities: list[str]) -> GraphPlan | None:
    """Build the small set of composed plans supported by the graph schema."""
    anchor = next((entity for entity in entities if entity[:1].isupper()), None)
    if not anchor:
        return None

    subclass_phrase = re.search(
        rf"(?:subclasses?\s+of\s+{re.escape(anchor)}|{re.escape(anchor)}\s+subclasses?)",
        query,
        re.IGNORECASE,
    )
    override_intent = re.search(
        r"\b(?:override|implement|provide|define)\w*\b",
        query,
        re.IGNORECASE,
    )
    if subclass_phrase and override_intent:
        callable_names = [match.group(1) for match in _CALLABLE_RE.finditer(query)]
        method_match = re.search(
            r"\b(?:override|implement|provide|define)\w*\s+"
            r"(?:(?:the|their)\s+)?(?:own\s+)?(?:private\s+)?"
            r"(?P<method>_?[A-Za-z][A-Za-z0-9_]*)",
            query,
            re.IGNORECASE,
        )
        terminal_name = method_match.group("method") if method_match else None
        if terminal_name in {"method", "methods"}:
            terminal_name = None
        if not terminal_name:
            terminal_name = next(
                (
                    entity
                    for entity in callable_names + entities
                    if entity != anchor and not entity[:1].isupper()
                ),
                None,
            )
        return GraphPlan(
            anchor=anchor,
            steps=(
                RelationStep("INHERITS", "in"),
                RelationStep("DEFINES", "out"),
            ),
            result_step=1,
            terminal_name=terminal_name,
        )

    if re.search(r"\bmethods?\b.*\binherit\w*\s+from\b", query, re.IGNORECASE):
        return GraphPlan(
            anchor=anchor,
            steps=(
                RelationStep("INHERITS", "out"),
                RelationStep("DEFINES", "out"),
            ),
            result_step=2,
        )

    return None


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
    if relation and (hop_match := _HOP_COUNT_RE.search(query)):
        raw_count = hop_match.group("count").lower()
        max_hops = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS[raw_count]

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
    for m in _CALLABLE_RE.finditer(query):
        _add(m.group(1))
    for m in _PASCAL_RE.finditer(query):
        name = m.group(1)
        # Short domain acronyms such as LLM and SQL are common prose, not
        # reliable exact graph anchors unless the user quotes them.
        if not (name.isupper() and len(name) <= 4):
            _add(name)
    for m in _SNAKE_RE.finditer(query):
        _add(m.group(1))

    graph_plan = _build_graph_plan(query, entities)
    if graph_plan:
        first_step = graph_plan.steps[0]
        relation = first_step.relation
        direction = first_step.direction
        max_hops = first_step.max_hops

    label_match = _TARGET_LABEL_RE.search(query)
    target_label = _TARGET_LABELS[label_match.group("kind").lower()] if label_match else None

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
        graph_plan=graph_plan,
        target_label=target_label,
    )
