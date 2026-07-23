from hybrid_rag.retrieval.skeletonizer import skeletonize_file


def test_skeletonize_python(tmp_path):
    code = """
class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def add(self, a, b):
        \"\"\"Add two numbers.\"\"\"
        return a + b

    def subtract(self, a, b):
        \"\"\"Subtract two numbers.\"\"\"
        return a - b
"""
    fpath = tmp_path / "calc.py"
    fpath.write_text(code, encoding="utf-8")

    # Focus on 'add'
    res = skeletonize_file(fpath, focus_names=["add"], language="python")
    assert "def add" in res
    assert "return a + b" in res
    assert "def subtract" in res
    assert "Subtract two numbers" in res
    assert "return a - b" not in res
    assert "..." in res


def test_skeletonize_python_no_match(tmp_path):
    code = """
def standalone():
    print("hello")
"""
    fpath = tmp_path / "standalone.py"
    fpath.write_text(code, encoding="utf-8")

    res = skeletonize_file(fpath, focus_names=["other"], language="python")
    assert 'print("hello")' not in res
    assert "..." in res
