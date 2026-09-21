import ast
from pathlib import Path


FERMION_DIR = Path(__file__).parents[1] / "src" / "edinpy" / "fermion"


def test_all_fermion_functions_and_methods_have_docstrings():
    missing = []
    for path in sorted(FERMION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.name}:{node.lineno}:{node.name}")
    assert missing == []
