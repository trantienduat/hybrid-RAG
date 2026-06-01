"""Quick smoke test for parser.py"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_rag.ingestion.parser import parse_file

r = parse_file(Path("src/hybrid_rag/ingestion/parser.py"), Path("."))
print(f"Nodes={len(r.nodes)} Edges={len(r.edges)} Errors={r.errors}")
print("Labels:", dict(Counter(n.label for n in r.nodes)))
fn = next((n for n in r.nodes if n.label == "Function"), None)
if fn:
    print("Sample fn:", fn.id)
    print("  sig:", fn.properties.get("signature", "?")[:70])
cls = next((n for n in r.nodes if n.label == "Class"), None)
if cls:
    print("Sample class:", cls.id)
