"""Spike 01: AST parser comparison — ast vs tree-sitter. Run from project root with venv active."""
import ast
import time
from pathlib import Path

import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# Use stdlib as corpus (always available, ~same complexity as real codebase)
import os, sysconfig
corpus_root = Path(sysconfig.get_path("stdlib"))
if not corpus_root or not corpus_root.exists():
    corpus_root = Path(os.__file__).parent

py_files = list(corpus_root.rglob("*.py"))[:100]
print(f"Corpus : {corpus_root}")
print(f"Files  : {len(py_files)}")
print()

# ── Approach 1: stdlib ast ─────────────────────────────────────────
errors_ast, times_ast = [], []
for f in py_files:
    src = f.read_text(encoding="utf-8", errors="ignore")
    t0 = time.perf_counter()
    try:
        ast.parse(src)
        times_ast.append(time.perf_counter() - t0)
    except SyntaxError as e:
        errors_ast.append(f.name)

# ── Approach 2: tree-sitter ────────────────────────────────────────
PY_LANG = Language(tspython.language())
ts_parser = Parser(PY_LANG)
errors_ts, times_ts = [], []
for f in py_files:
    src = f.read_bytes()
    t0 = time.perf_counter()
    tree = ts_parser.parse(src)
    times_ts.append(time.perf_counter() - t0)
    if tree.root_node.has_error:
        errors_ts.append(f.name)

# ── Node extraction demo (tree-sitter) ────────────────────────────
sample = py_files[0]
src_bytes = sample.read_bytes()
tree = ts_parser.parse(src_bytes)

def node_text(node):
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

classes = [
    node_text(n.child_by_field_name("name"))
    for n in tree.root_node.children
    if n.type == "class_definition" and n.child_by_field_name("name")
]
fns = [
    node_text(n.child_by_field_name("name"))
    for n in tree.root_node.children
    if n.type == "function_definition" and n.child_by_field_name("name")
]

# ── Java support check ─────────────────────────────────────────────
JAVA_LANG = Language(tsjava.language())
java_parser = Parser(JAVA_LANG)
java_sample = b"public class Foo { public void bar() {} }"
java_tree = java_parser.parse(java_sample)

# ── Results ────────────────────────────────────────────────────────
W = 55
print("=" * W)
print(f"{'Metric':<30} {'ast':>10} {'tree-sitter':>12}")
print("=" * W)
print(f"{'Files parsed':<30} {len(times_ast):>10} {len(times_ts):>12}")
print(f"{'Parse errors':<30} {len(errors_ast):>10} {len(errors_ts):>12}")
print(f"{'Total time (s)':<30} {sum(times_ast):>10.3f} {sum(times_ts):>12.3f}")
print(f"{'Avg ms/file':<30} {sum(times_ast)/max(len(times_ast),1)*1000:>10.2f} {sum(times_ts)/max(len(times_ts),1)*1000:>12.2f}")
print(f"{'Java support':<30} {'No':>10} {'Yes':>12}")
print(f"{'Multi-language':<30} {'No':>10} {'Yes':>12}")
print("=" * W)
print()
print(f"Sample file : {sample.name}")
print(f"  Classes   : {classes or ['(none)']}")
print(f"  Functions : {fns[:5] or ['(none)']}")
print()
print(f"Java parse OK: {not java_tree.root_node.has_error}")

# ── Verdict ────────────────────────────────────────────────────────
print()
print("VERDICT:")
if len(errors_ts) <= len(errors_ast):
    print("  tree-sitter: fewer/equal errors, Java support → PREFERRED")
else:
    print("  ast: fewer errors → reconsider")
