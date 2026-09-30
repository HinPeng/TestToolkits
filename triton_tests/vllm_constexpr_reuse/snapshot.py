"""Verify the frozen JIT definitions without importing torch, Triton or vLLM."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def definitions(path):
    source = Path(path).read_text()
    lines = source.splitlines(keepends=True)
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result[node.name] = (start, "".join(lines[start - 1:node.end_lineno]))
    return result


def verify(upstream=None):
    manifest = json.loads((ROOT / "sources.json").read_text())
    frozen = definitions(ROOT / "kernels.py")
    for name, entry in manifest["functions"].items():
        code = frozen[name][1]
        assert hashlib.sha256(code.encode()).hexdigest() == entry["sha256"], name
        if upstream is not None:
            original = definitions(Path(upstream) / entry["path"])[name][1]
            assert code == original, f"upstream definition changed: {name}"
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="vllm-ascend checkout to compare")
    args = parser.parse_args()
    manifest = verify(args.upstream)
    print(f"Verified {len(manifest['functions'])} JIT definitions; baseline {manifest['revision']}")
