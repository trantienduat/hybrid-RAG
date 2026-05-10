"""
AST-based source code parser using tree-sitter.

Extracts raw node data (modules, classes, functions, variables) from Python
and Java files. Output is a list of NodeData and EdgeData dicts that match
the KG schema in docs/schema/kg-schema.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

# ── Language setup ────────────────────────────────────────────────
_PY_LANG = Language(tspython.language())
_JAVA_LANG = Language(tsjava.language())

_PARSERS: dict[str, Parser] = {
    "python": Parser(_PY_LANG),
    "java": Parser(_JAVA_LANG),
}

LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".java": "java",
}


# ── Data models ───────────────────────────────────────────────────

@dataclass
class NodeData:
    """Represents a KG node to be written to FalkorDB."""
    label: str          # Module | Class | Function | Variable
    id: str             # Unique key (see schema)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeData:
    """Represents a KG edge to be written to FalkorDB."""
    src_id: str
    rel: str            # IMPORTS | DEFINES | INHERITS | CALLS | USES | DEFINED_IN
    dst_id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    nodes: list[NodeData] = field(default_factory=list)
    edges: list[EdgeData] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────

def parse_file(file_path: Path, repo_root: Path) -> ParseResult:
    """Parse a single source file and return nodes + edges."""
    ext = file_path.suffix.lower()
    language = LANGUAGE_BY_EXT.get(ext)
    if language is None:
        return ParseResult(errors=[f"Unsupported extension: {ext}"])

    try:
        src_bytes = file_path.read_bytes()
    except OSError as e:
        return ParseResult(errors=[f"Cannot read {file_path}: {e}"])

    parser = _PARSERS[language]
    tree = parser.parse(src_bytes)

    rel_path = str(file_path.relative_to(repo_root))

    if language == "python":
        return _extract_python(tree.root_node, src_bytes, rel_path)
    return ParseResult(errors=[f"Java extraction not yet implemented: {rel_path}"])


def parse_repo(repo_root: Path, languages: list[str] | None = None) -> ParseResult:
    """Parse all supported source files under repo_root."""
    languages = languages or ["python"]
    exts = {ext for ext, lang in LANGUAGE_BY_EXT.items() if lang in languages}

    combined = ParseResult()
    for ext in exts:
        for fpath in sorted(repo_root.rglob(f"*{ext}")):
            result = parse_file(fpath, repo_root)
            combined.nodes.extend(result.nodes)
            combined.edges.extend(result.edges)
            combined.errors.extend(result.errors)
    return combined


# ── Python extraction ─────────────────────────────────────────────

def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_text(node: Node, field_name: str, src: bytes) -> str | None:
    child = node.child_by_field_name(field_name)
    return _text(child, src) if child else None


def _docstring(node: Node, src: bytes) -> str | None:
    """Extract first string literal if it's the first statement (docstring convention)."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for s in stmt.children:
                        if s.type == "string":
                            raw = _text(s, src)
                            # Strip quotes
                            return re.sub(r'^["\' ]{1,3}|["\' ]{1,3}$', '', raw).strip()
                break  # only check first statement
    return None


def _extract_python(root: Node, src: bytes, rel_path: str) -> ParseResult:
    result = ParseResult()
    module_name = Path(rel_path).stem
    module_id = rel_path

    # Determine module type
    if rel_path.endswith("__init__.py"):
        mod_type = "init"
    elif "/test" in rel_path or rel_path.startswith("test"):
        mod_type = "test"
    else:
        mod_type = "source"

    module_node = NodeData(
        label="Module",
        id=module_id,
        properties={
            "name": module_name,
            "file_path": rel_path,
            "language": "python",
            "type": mod_type,
            "line_count": root.end_point[0] + 1,
        },
    )
    result.nodes.append(module_node)

    # Walk top-level children
    for node in root.children:
        if node.type == "import_statement":
            _handle_import(node, src, module_id, result)
        elif node.type == "import_from_statement":
            _handle_from_import(node, src, module_id, result)
        elif node.type == "class_definition":
            _handle_class(node, src, rel_path, module_id, result)
        elif node.type == "function_definition":
            _handle_function(node, src, rel_path, module_id, class_name=None, result=result)

    return result


def _handle_import(node: Node, src: bytes, module_id: str, result: ParseResult) -> None:
    """import foo, import foo as bar"""
    for child in node.children:
        if child.type in ("dotted_name", "aliased_import"):
            name_node = child.child_by_field_name("name") or child
            name = _text(name_node, src).split(".")[0]
            dst_id = name  # external stub id
            _ensure_stub(name, result)
            result.edges.append(EdgeData(
                src_id=module_id,
                rel="IMPORTS",
                dst_id=dst_id,
                properties={"is_from": False},
            ))


def _handle_from_import(node: Node, src: bytes, module_id: str, result: ParseResult) -> None:
    """from foo.bar import baz"""
    module_part = None
    for child in node.children:
        if child.type == "dotted_name" and module_part is None:
            module_part = _text(child, src)
            break
        if child.type == "relative_import":
            module_part = _text(child, src)
            break

    if module_part:
        base = module_part.lstrip(".").split(".")[0]
        if base:
            _ensure_stub(base, result)
            result.edges.append(EdgeData(
                src_id=module_id,
                rel="IMPORTS",
                dst_id=base,
                properties={"is_from": True},
            ))


def _handle_class(
    node: Node, src: bytes, rel_path: str, module_id: str, result: ParseResult
) -> None:
    name = _child_text(node, "name", src) or "UnknownClass"
    class_id = f"{rel_path}::{name}"

    bases: list[str] = []
    arg_list = node.child_by_field_name("superclasses")
    if arg_list:
        for arg in arg_list.children:
            if arg.type in ("identifier", "attribute"):
                bases.append(_text(arg, src))

    is_abstract = any(
        _text(d, src) in ("ABC", "ABCMeta", "abstractmethod")
        for d in node.children
        if d.type == "decorator"
    )

    class_node = NodeData(
        label="Class",
        id=class_id,
        properties={
            "name": name,
            "file_path": rel_path,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "docstring": _docstring(node, src),
            "is_abstract": is_abstract,
        },
    )
    result.nodes.append(class_node)

    # Module DEFINES Class
    result.edges.append(EdgeData(src_id=module_id, rel="DEFINES", dst_id=class_id))
    # Class DEFINED_IN Module
    result.edges.append(EdgeData(src_id=class_id, rel="DEFINED_IN", dst_id=module_id))

    # Inheritance edges (stub targets resolved later by entity resolver)
    for order, base in enumerate(bases, start=1):
        base_stub_id = base.split(".")[-1]  # use simple name as stub
        _ensure_stub(base_stub_id, result)
        result.edges.append(EdgeData(
            src_id=class_id,
            rel="INHERITS",
            dst_id=base_stub_id,
            properties={"order": order},
        ))

    # Extract methods from class body
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "function_definition":
                _handle_function(child, src, rel_path, module_id, class_name=name, result=result)


def _handle_function(
    node: Node,
    src: bytes,
    rel_path: str,
    module_id: str,
    class_name: str | None,
    result: ParseResult,
) -> None:
    name = _child_text(node, "name", src) or "unknown"
    fn_id = f"{rel_path}::{class_name}::{name}" if class_name else f"{rel_path}::{name}"
    parent_id = f"{rel_path}::{class_name}" if class_name else module_id

    is_async = any(c.type == "async" for c in node.children)
    is_abstract = any(
        "abstractmethod" in _text(d, src)
        for d in node.children
        if d.type == "decorator"
    )
    is_property = any(
        "property" in _text(d, src)
        for d in node.children
        if d.type == "decorator"
    )

    params_node = node.child_by_field_name("parameters")
    signature = f"def {name}{_text(params_node, src) if params_node else '()'}"

    fn_node = NodeData(
        label="Function",
        id=fn_id,
        properties={
            "name": name,
            "file_path": rel_path,
            "class_name": class_name,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "signature": signature,
            "docstring": _docstring(node, src),
            "is_async": is_async,
            "is_abstract": is_abstract,
            "is_property": is_property,
        },
    )
    result.nodes.append(fn_node)

    # Parent DEFINES Function
    result.edges.append(EdgeData(src_id=parent_id, rel="DEFINES", dst_id=fn_id))
    # Function DEFINED_IN Module
    result.edges.append(EdgeData(src_id=fn_id, rel="DEFINED_IN", dst_id=module_id))

    # Extract call sites from function body
    body = node.child_by_field_name("body")
    if body:
        _extract_calls(body, src, fn_id, result)


def _extract_calls(body: Node, src: bytes, caller_id: str, result: ParseResult) -> None:
    """Walk body for call expressions and emit CALLS edges (stub targets)."""
    for child in body.children:
        _walk_calls(child, src, caller_id, result)


def _walk_calls(node: Node, src: bytes, caller_id: str, result: ParseResult) -> None:
    if node.type == "call":
        fn_node = node.child_by_field_name("function")
        if fn_node:
            callee_text = _text(fn_node, src)
            # Simple name: `foo()` or `self.foo()` → use last segment
            callee_name = callee_text.split(".")[-1]
            stub_id = f"__call__{callee_name}"
            result.edges.append(EdgeData(
                src_id=caller_id,
                rel="CALLS",
                dst_id=stub_id,
                properties={"line": node.start_point[0] + 1, "callee_expr": callee_text},
            ))
    for child in node.children:
        _walk_calls(child, src, caller_id, result)


def _ensure_stub(name: str, result: ParseResult) -> None:
    """Add external stub node if not already present."""
    stub_id = name
    if not any(n.id == stub_id for n in result.nodes):
        result.nodes.append(NodeData(
            label="Module",
            id=stub_id,
            properties={
                "name": name,
                "file_path": "",
                "language": "python",
                "type": "external",
                "line_count": 0,
            },
        ))
