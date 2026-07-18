from __future__ import annotations

import logging
import re
from pathlib import Path
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

logger = logging.getLogger(__name__)

# Setup tree-sitter languages
_PY_LANG = Language(tspython.language())
_JAVA_LANG = Language(tsjava.language())

_PARSERS: dict[str, Parser] = {
    "python": Parser(_PY_LANG),
    "java": Parser(_JAVA_LANG),
}

def _text(node: Node, src: bytes) -> str:
    """Helper to extract and decode text from a tree-sitter node."""
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

def _docstring(node: Node, src: bytes, language: str) -> str | None:
    """Extract docstring/javadoc comment for a python/java definition node."""
    if language == "python":
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for s in stmt.children:
                            if s.type == "string":
                                return _text(s, src).strip()
                    break  # docstring is only first stmt
    elif language == "java":
        parent = node.parent
        if parent:
            idx = node.sibling_index
            if idx is not None and idx > 0:
                prev_sibling = parent.children[idx - 1]
                if prev_sibling.type == "block_comment" and _text(prev_sibling, src).startswith("/**"):
                    return _text(prev_sibling, src).strip()
    return None

def skeletonize_file(
    file_path: Path,
    focus_names: list[str],
    language: str = "python"
) -> str:
    """
    Read file_path, parse AST, and return a skeletonized code representation.
    The focus_names are preserved in full, while all other methods/functions
    in the same scopes are collapsed to their signatures and docstrings.
    """
    if not file_path.exists():
        return ""

    try:
        src = file_path.read_bytes()
    except OSError as e:
        logger.error("Failed to read file %s for skeletonization: %s", file_path, e)
        return ""

    parser = _PARSERS.get(language)
    if not parser:
        return src.decode("utf-8", errors="replace")

    tree = parser.parse(src)
    root = tree.root_node
    replacements = []

    def traverse(node: Node, in_class_name: str | None = None) -> None:
        if language == "python":
            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                class_name = _text(name_node, src) if name_node else "Unknown"
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        traverse(child, in_class_name=class_name)
            elif node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                func_name = _text(name_node, src) if name_node else "Unknown"

                # Check if this function is a focus target
                is_focus = False
                for focus in focus_names:
                    if focus == func_name:
                        is_focus = True
                        break
                    if in_class_name and (focus == f"{in_class_name}.{func_name}" or focus == func_name):
                        is_focus = True
                        break

                if not is_focus:
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        # Extract signature (everything up to body block)
                        sig_text = src[node.start_byte : body_node.start_byte].decode("utf-8", errors="replace").rstrip()
                        doc = _docstring(node, src, "python")

                        # Determine indentation of body
                        indent = "    "
                        body_lines = _text(body_node, src).splitlines()
                        if len(body_lines) > 1:
                            first_line = body_lines[1]
                            indent_match = re.match(r"^(\s+)", first_line)
                            if indent_match:
                                indent = indent_match.group(1)

                        parts = [sig_text]
                        if doc:
                            parts.append(f"\n{indent}{doc}")
                        parts.append(f"\n{indent}...")
                        replacements.append((node.start_byte, node.end_byte, "".join(parts)))
                return

        elif language == "java":
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                class_name = _text(name_node, src) if name_node else "Unknown"
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        traverse(child, in_class_name=class_name)
            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                method_name = _text(name_node, src) if name_node else "Unknown"

                is_focus = False
                for focus in focus_names:
                    if focus == method_name:
                        is_focus = True
                        break
                    if in_class_name and (focus == f"{in_class_name}.{method_name}" or focus == method_name):
                        is_focus = True
                        break

                if not is_focus:
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        # Signature (everything up to body block)
                        sig_text = src[node.start_byte : body_node.start_byte].decode("utf-8", errors="replace").rstrip()
                        doc = _docstring(node, src, "java")

                        # Indentation
                        indent = "    "
                        body_lines = _text(body_node, src).splitlines()
                        if len(body_lines) > 1:
                            first_line = body_lines[1]
                            indent_match = re.match(r"^(\s+)", first_line)
                            if indent_match:
                                indent = indent_match.group(1)

                        parts = [sig_text]
                        if doc:
                            parts.append(f"\n{indent}{doc}")
                        parts.append(f"\n{indent}{{ ... }}")
                        replacements.append((node.start_byte, node.end_byte, "".join(parts)))
                return

        for child in node.children:
            traverse(child, in_class_name)

    traverse(root)

    # Sort replacements by start_byte descending to replace from end to beginning
    replacements.sort(key=lambda x: x[0], reverse=True)

    result_bytes = bytearray(src)
    for start, end, text in replacements:
        result_bytes[start:end] = text.encode("utf-8")

    return result_bytes.decode("utf-8", errors="replace")
