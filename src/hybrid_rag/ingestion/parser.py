"""
AST-based source code parser using tree-sitter.

Extracts raw node data (modules, classes, functions, variables) from Python
and Java files. Output is NodeData and EdgeData dataclasses that match
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

# Mapping of file extensions to their respective languages
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".java": "java",
}


# ── Data models ───────────────────────────────────────────────────


@dataclass
class NodeData:
    """
    Represents a Knowledge Graph (KG) node to be written to FalkorDB.

    Attributes:
        label: The type of node (e.g., Module, Class, Function, Variable).
        id: Unique identifier for the node, following the schema defined in docs/schema/kg-schema.md.
        properties: A dictionary of key-value pairs representing node attributes.
    """

    label: str  # Module | Class | Function | Variable
    id: str  # Unique key (see schema)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeData:
    """
    Represents a Knowledge Graph (KG) edge/relationship to be written to FalkorDB.

    Attributes:
        src_id: The ID of the source node.
        rel: The relationship type (e.g., IMPORTS, DEFINES, CALLS).
        dst_id: The ID of the destination node.
        properties: A dictionary of key-value pairs representing edge attributes (e.g., line numbers).
    """

    src_id: str
    rel: str  # IMPORTS | DEFINES | INHERITS | CALLS | USES | DEFINED_IN
    dst_id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """
    Container for the results of a parsing operation.

    Attributes:
        nodes: List of discovered NodeData objects.
        edges: List of discovered EdgeData objects linking the nodes.
        errors: List of error messages encountered during parsing.
    """

    nodes: list[NodeData] = field(default_factory=list)
    edges: list[EdgeData] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def rel_path_to_fqn(rel_path: str) -> str:
    """
    Convert a repository-relative file path to a Fully Qualified Name (FQN) module name.

    Examples:
        src/hybrid_rag/ports/embedder.py -> hybrid_rag.ports.embedder
        src/hybrid_rag/__init__.py -> hybrid_rag
        tests/unit/test_retrieval.py -> tests.unit.test_retrieval
    """
    p = rel_path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    elif p.endswith(".java"):
        p = p[:-5]

    if p.endswith("/__init__"):
        p = p[:-9]

    parts = p.split("/")
    if len(parts) >= 3 and parts[:3] == ["src", "main", "java"]:
        parts = parts[3:]
    elif len(parts) >= 3 and parts[:3] == ["src", "test", "java"]:
        parts = parts[3:]
    elif parts and parts[0] == "src":
        parts = parts[1:]

    return ".".join(parts)


# ── Public API ────────────────────────────────────────────────────


def parse_file(file_path: Path, repo_root: Path, repo_name: str = "") -> ParseResult:
    """
    Parse a single source file and return its graph representation (nodes + edges).

    Args:
        file_path: The absolute path to the file to be parsed.
        repo_root: The root directory of the repository, used to calculate relative paths for node IDs.
        repo_name: The custom namespace name for the repository.

    Returns:
        A ParseResult containing the extracted nodes, edges, and any errors.
    """
    ext = file_path.suffix.lower()
    if file_path.name == "Dockerfile":
        ext = "Dockerfile"

    # Non-code configuration and document files handling
    if ext in (".yaml", ".yml", ".md", "Dockerfile"):
        rel_path = str(file_path.relative_to(repo_root))
        label = "Document" if ext == ".md" else "Configuration"
        node_id = f"{repo_name}::{rel_path}"

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ParseResult(errors=[f"Cannot read {file_path}: {e}"])

        node = NodeData(
            label=label,
            id=node_id,
            properties={
                "name": file_path.name,
                "file_path": rel_path,
                "repository": repo_name,
                "text": content,
            }
        )

        parent_dir = str(file_path.parent.relative_to(repo_root))
        if parent_dir == ".":
            dst_id = repo_name
            dst_label = "RepositoryMetadata"
        else:
            dst_id = f"{repo_name}::dir::{parent_dir}"
            dst_label = "Directory"

        edge = EdgeData(
            src_id=node_id,
            rel="WEAK_LINK",
            dst_id=dst_id,
            properties={"repository": repo_name}
        )

        nodes_to_add = [node]
        edges_to_add = [edge]

        if dst_label == "Directory":
            dir_node = NodeData(
                label="Directory",
                id=dst_id,
                properties={
                    "name": file_path.parent.name,
                    "path": parent_dir,
                    "repository": repo_name
                }
            )
            nodes_to_add.append(dir_node)

            current_path = Path(parent_dir)
            while current_path.parent != Path("."):
                curr_dir_id = f"{repo_name}::dir::{current_path}"
                parent_dir_path = current_path.parent
                parent_dir_id = f"{repo_name}::dir::{parent_dir_path}"

                p_dir_node = NodeData(
                    label="Directory",
                    id=parent_dir_id,
                    properties={
                        "name": parent_dir_path.name,
                        "path": str(parent_dir_path),
                        "repository": repo_name
                    }
                )
                nodes_to_add.append(p_dir_node)

                p_edge = EdgeData(
                    src_id=curr_dir_id,
                    rel="WEAK_LINK",
                    dst_id=parent_dir_id,
                    properties={"repository": repo_name}
                )
                edges_to_add.append(p_edge)
                current_path = parent_dir_path

            top_dir_id = f"{repo_name}::dir::{current_path}"
            root_edge = EdgeData(
                src_id=top_dir_id,
                rel="WEAK_LINK",
                dst_id=repo_name,
                properties={"repository": repo_name}
            )
            edges_to_add.append(root_edge)

        return ParseResult(nodes=nodes_to_add, edges=edges_to_add)

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
        res = _extract_python(tree.root_node, src_bytes, rel_path)
        # Assign repository property to all nodes in the file
        for node in res.nodes:
            node.properties["repository"] = repo_name
        return res
    return ParseResult(errors=[f"Java extraction not yet implemented: {rel_path}"])


def parse_repo(
    repo_root: Path,
    languages: list[str] | None = None,
    repo_name: str = "",
    excludes: list[str] | None = None,
) -> ParseResult:
    """
    Recursively parse all supported source files under the given repository root.

    Args:
        repo_root: The root directory of the repository to scan.
        languages: List of languages to include (e.g., ["python", "java"]). Defaults to ["python"].
        repo_name: The custom namespace name for the repository.
        excludes: List of folder/file name patterns to exclude from parsing.
    Returns:
        A combined ParseResult containing graph data from all parsed files.
    """
    languages = languages or ["python"]
    exts = {ext for ext, lang in LANGUAGE_BY_EXT.items() if lang in languages}

    exclude_set = (
        set(excludes)
        if excludes is not None
        else {
            ".venv",
            "venv",
            "fixtures",
            "experiments",
            "dist",
            "build",
            ".git",
            "__pycache__",
            "node_modules",
            ".agents",
            ".gemini",
            ".pytest_cache",
            ".ruff_cache",
            ".roo",
            ".clinerules",
        }
    )

    import os
    non_code_exts = set()
    if os.environ.get("NON_CODE_INGESTION", "false").lower() == "true":
        non_code_exts = {".yaml", ".yml", ".md", "Dockerfile"}

    # 1. Collect all valid files to parse
    files_to_parse: list[tuple[Path, str]] = []
    
    # Track directories to build structure
    seen_dirs: set[str] = set()
    dir_nodes: list[NodeData] = []
    dir_edges: list[EdgeData] = []

    def process_fpath_dirs(fpath: Path):
        try:
            rel_parts = fpath.relative_to(repo_root).parts
        except ValueError:
            return
        
        # Walk up the directory hierarchy for this file
        current = Path(*rel_parts[:-1]) if len(rel_parts) > 1 else None
        while current and str(current) != ".":
            dir_str = str(current)
            if dir_str in seen_dirs:
                break
            seen_dirs.add(dir_str)

            dir_id = f"{repo_name}::dir::{dir_str}"
            dir_nodes.append(NodeData(
                label="Directory",
                id=dir_id,
                properties={
                    "name": current.name,
                    "path": dir_str,
                    "repository": repo_name,
                }
            ))

            parent = current.parent
            if str(parent) == ".":
                dir_edges.append(EdgeData(
                    src_id=dir_id,
                    rel="WEAK_LINK",
                    dst_id=repo_name,
                    properties={"repository": repo_name},
                ))
            else:
                parent_id = f"{repo_name}::dir::{parent}"
                dir_edges.append(EdgeData(
                    src_id=dir_id,
                    rel="WEAK_LINK",
                    dst_id=parent_id,
                    properties={"repository": repo_name},
                ))
            current = parent if str(parent) != "." else None

    # Code files
    for ext in exts:
        for fpath in repo_root.rglob(f"*{ext}"):
            try:
                rel_parts = fpath.relative_to(repo_root).parts
            except ValueError:
                continue
            if any(p in exclude_set or p.startswith(".venv") for p in rel_parts):
                continue
            files_to_parse.append((fpath, repo_name))
            process_fpath_dirs(fpath)

    # Non-code files
    for ext in non_code_exts:
        pattern = "Dockerfile" if ext == "Dockerfile" else f"*{ext}"
        for fpath in repo_root.rglob(pattern):
            try:
                rel_parts = fpath.relative_to(repo_root).parts
            except ValueError:
                continue
            if any(p in exclude_set or p.startswith(".venv") for p in rel_parts):
                continue
            files_to_parse.append((fpath, repo_name))
            process_fpath_dirs(fpath)

    # Sort to keep order deterministic
    files_to_parse.sort(key=lambda x: x[0])

    combined = ParseResult()

    # 2. Parse files concurrently using ThreadPoolExecutor
    # tree-sitter C bindings release the GIL, and most time is spent in IO and tree-sitter parsing
    from concurrent.futures import ThreadPoolExecutor
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    
    def parse_single_file(arg: tuple[Path, str]) -> ParseResult:
        fpath, rname = arg
        return parse_file(fpath, repo_root, repo_name=rname)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(parse_single_file, files_to_parse)
        for res in results:
            combined.nodes.extend(res.nodes)
            combined.edges.extend(res.edges)
            combined.errors.extend(res.errors)

    # Add directory nodes and edges
    combined.nodes.extend(dir_nodes)
    combined.edges.extend(dir_edges)

    # ── Link Directory → Module via DEFINES edges ──────────────────────────────
    # _extract_python creates Module nodes but doesn't wire them to Directory nodes.
    # We do it here, post-parse, based on each Module's file_path property.
    for node in combined.nodes:
        if node.label != "Module":
            continue
        file_path_str = node.properties.get("file_path", "")
        if not file_path_str:
            continue
        parent_dir = str(Path(file_path_str).parent)
        if parent_dir == ".":
            # Top-level file: link to RepositoryMetadata
            combined.edges.append(EdgeData(
                src_id=repo_name,
                rel="DEFINES",
                dst_id=node.id,
                properties={"repository": repo_name},
            ))
        else:
            dir_id = f"{repo_name}::dir::{parent_dir}"
            combined.edges.append(EdgeData(
                src_id=dir_id,
                rel="DEFINES",
                dst_id=node.id,
                properties={"repository": repo_name},
            ))

    return combined


# ── Python extraction ─────────────────────────────────────────────


def _text(node: Node, src: bytes) -> str:
    """Helper to extract and decode text from a tree-sitter node."""
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _child_text(node: Node, field_name: str, src: bytes) -> str | None:
    """Helper to extract text from a specific child node identified by field name."""
    child = node.child_by_field_name(field_name)
    return _text(child, src) if child else None


def _docstring(node: Node, src: bytes) -> str | None:
    """
    Extract the docstring of a Python block (class/function/module).
    Following Python convention, it looks for the first statement if it is a string literal.
    """
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for s in stmt.children:
                        if s.type == "string":
                            raw = _text(s, src)
                            # Strip quotes (triple or single)
                            return re.sub(r'^["\' ]{1,3}|["\' ]{1,3}$', "", raw).strip()
                break  # only check first statement
    return None


def _extract_python(root: Node, src: bytes, rel_path: str) -> ParseResult:
    """
    Internal driver for Python file extraction.
    Walks the AST top-level and delegates to specific handlers.
    """
    result = ParseResult()
    module_fqn = rel_path_to_fqn(rel_path)
    module_id = module_fqn

    # Determine module type based on path conventions
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
            "name": module_fqn.split(".")[-1],
            "file_path": rel_path,
            "language": "python",
            "type": mod_type,
            "line_count": root.end_point[0] + 1,
        },
    )
    result.nodes.append(module_node)

    # Walk top-level children in the module
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
    """Handles `import foo` or `import foo as bar` statements."""
    for child in node.children:
        if child.type in ("dotted_name", "aliased_import"):
            name_node = child.child_by_field_name("name") or child
            name = _text(name_node, src).split(".")[0]
            dst_id = name  # external stub id reference
            _ensure_stub(name, result)
            result.edges.append(
                EdgeData(
                    src_id=module_id,
                    rel="IMPORTS",
                    dst_id=dst_id,
                    properties={"is_from": False},
                )
            )


def _handle_from_import(node: Node, src: bytes, module_id: str, result: ParseResult) -> None:
    """Handles `from foo import bar` statements."""
    module_part = None
    for child in node.children:
        if child.type == "dotted_name" and module_part is None:
            module_part = _text(child, src)
            break
        if child.type == "relative_import":
            module_part = _text(child, src)
            break

    if module_part:
        # We use the base package name for external dependency tracking
        base = module_part.lstrip(".").split(".")[0]
        if base:
            _ensure_stub(base, result)
            result.edges.append(
                EdgeData(
                    src_id=module_id,
                    rel="IMPORTS",
                    dst_id=base,
                    properties={"is_from": True},
                )
            )


def _handle_class(
    node: Node, src: bytes, rel_path: str, module_id: str, result: ParseResult
) -> None:
    """Extracts Class node data and its members (methods)."""
    name = _child_text(node, "name", src) or "UnknownClass"
    class_id = f"{module_id}.{name}"

    # Extract base classes (inheritance)
    bases: list[str] = []
    arg_list = node.child_by_field_name("superclasses")
    if arg_list:
        for arg in arg_list.children:
            if arg.type in ("identifier", "attribute"):
                bases.append(_text(arg, src))

    # Check for abstract markers
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

    # Establish ownership relationships
    # Module DEFINES Class
    result.edges.append(EdgeData(src_id=module_id, rel="DEFINES", dst_id=class_id))
    # Class DEFINED_IN Module
    result.edges.append(EdgeData(src_id=class_id, rel="DEFINED_IN", dst_id=module_id))

    # Inheritance edges (stub targets resolved later by entity resolver)
    for order, base in enumerate(bases, start=1):
        base_stub_id = base.split(".")[-1]  # use simple name as stub
        _ensure_stub(base_stub_id, result)
        result.edges.append(
            EdgeData(
                src_id=class_id,
                rel="INHERITS",
                dst_id=base_stub_id,
                properties={"order": order},
            )
        )

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
    """Extracts Function/Method node data and its call sites."""
    name = _child_text(node, "name", src) or "unknown"
    parent_id = f"{module_id}.{class_name}" if class_name else module_id
    fn_id = f"{parent_id}.{name}"

    # Metadata extraction
    is_async = any(c.type == "async" for c in node.children)
    is_abstract = any(
        "abstractmethod" in _text(d, src) for d in node.children if d.type == "decorator"
    )
    is_property = any("property" in _text(d, src) for d in node.children if d.type == "decorator")

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

    # Parent (Module or Class) DEFINES Function
    result.edges.append(EdgeData(src_id=parent_id, rel="DEFINES", dst_id=fn_id))
    # Function DEFINED_IN Module
    result.edges.append(EdgeData(src_id=fn_id, rel="DEFINED_IN", dst_id=module_id))

    # Extract call sites from function body
    body = node.child_by_field_name("body")
    if body:
        _extract_calls(body, src, fn_id, result)


def _extract_calls(body: Node, src: bytes, caller_id: str, result: ParseResult) -> None:
    """Helper to initiate walking the function body for call sites."""
    for child in body.children:
        _walk_calls(child, src, caller_id, result)


def _walk_calls(node: Node, src: bytes, caller_id: str, result: ParseResult) -> None:
    """Recursively search for `call` nodes in the AST and emit CALLS edges."""
    if node.type == "call":
        fn_node = node.child_by_field_name("function")
        if fn_node:
            callee_text = _text(fn_node, src)
            # Simple name: `foo()` or `self.foo()` → use last segment as a stub ID
            callee_name = callee_text.split(".")[-1]
            stub_id = f"__call__{callee_name}"
            result.edges.append(
                EdgeData(
                    src_id=caller_id,
                    rel="CALLS",
                    dst_id=stub_id,
                    properties={"line": node.start_point[0] + 1, "callee_expr": callee_text},
                )
            )
    for child in node.children:
        _walk_calls(child, src, caller_id, result)


def _ensure_stub(name: str, result: ParseResult) -> None:
    """
    Ensure an external stub node exists in the result.
    Stubs are placeholders for entities defined outside the current file.
    """
    stub_id = name
    if not any(n.id == stub_id for n in result.nodes):
        result.nodes.append(
            NodeData(
                label="Module",
                id=stub_id,
                properties={
                    "name": name,
                    "file_path": "",
                    "language": "python",
                    "type": "external",
                    "line_count": 0,
                },
            )
        )
