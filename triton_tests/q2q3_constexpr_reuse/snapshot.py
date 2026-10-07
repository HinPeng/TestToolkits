"""Verify frozen Q2/Q3 JIT definitions without importing Triton or torch."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from analysis_adapter import jit_decorator
from cases import CASES

ROOT = Path(__file__).resolve().parent


def definitions(path, names=None):
    source = Path(path).read_text(encoding="utf-8-sig")
    lines = source.splitlines(keepends=True)
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if names is not None and node.name not in names:
            continue
        decorator = jit_decorator(node)
        if decorator is None:
            continue
        if node.name in result:
            raise AssertionError(f"Duplicate JIT definition: {node.name}")
        result[node.name] = (
            "".join(lines[decorator.lineno - 1:decorator.end_lineno])
            + "".join(lines[node.lineno - 1:node.end_lineno])
        )
    return result


def verify(upstream=None):
    manifest = json.loads((ROOT / "sources.json").read_text())
    frozen = definitions(ROOT / "kernels.py")
    assert set(frozen) == set(manifest["functions"]), "Snapshot inventory changed"
    for case in CASES:
        entry = manifest["functions"][case.kernel]
        assert entry["path"] == f"{case.collection}/src/kernels/{case.path}", case.id
    originals = {}
    for name, entry in manifest["functions"].items():
        code = frozen[name]
        assert hashlib.sha256(code.encode()).hexdigest() == entry["sha256"], name
        if upstream is not None:
            path = Path(upstream) / entry["path"]
            if path not in originals:
                names = {name for name, item in manifest["functions"].items()
                         if item["path"] == entry["path"]}
                originals[path] = definitions(path, names)
            assert code == originals[path][name], f"Upstream definition changed: {name}"
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="TritonAscendTest checkout to compare")
    args = parser.parse_args()
    manifest = verify(args.upstream)
    print(f"Verified {len(manifest['functions'])} Q2/Q3 JIT definitions")
