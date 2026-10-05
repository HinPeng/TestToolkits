"""Analyze Q2/Q3 constexpr categories on the CPU; exit 1 on oracle differences."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from analysis_adapter import Pointer, analyzer
from cases import CASES, RUNTIME, SCHEDULE


def find_root(explicit, env, name):
    supplied = explicit or os.environ.get(env)
    if supplied:
        result = Path(supplied).expanduser().resolve()
        if not result.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {result}")
        return result
    origins = (Path.cwd(), Path(__file__).resolve().parent)
    for origin in origins:
        for base in (origin, *origin.parents):
            for candidate in (base / name, base / "Ascend" / name):
                if candidate.is_dir():
                    return candidate.resolve()
    raise FileNotFoundError(f"Cannot locate {name}; set {env} or pass its root option")


def bind_arguments(api, fn, arguments):
    values = {}
    for name, value in arguments.items():
        if isinstance(value, dict):
            if set(value) == {"pointer"}:
                value = Pointer(value["pointer"])
            elif set(value) == {"dtype"}:
                value = vars(api.loader.language)[value["dtype"]]
            else:
                raise ValueError(f"Unsupported argument descriptor: {name}={value}")
        values[name] = value
    bound = fn.signature.bind(**values)
    bound.apply_defaults()
    return bound.arguments


def analyze_case(api, kernel_root, case):
    path = Path(kernel_root) / case.collection / "src/kernels" / case.path
    fn = api.loader.kernel(path, case.kernel)
    names = {p.name for p in fn.params if p.is_constexpr}
    if names != set(case.expected):
        raise AssertionError(f"{case.id}: constexpr inventory changed: {names ^ set(case.expected)}")
    before = fn.src, tuple(fn.arg_names), fn.constexprs
    bound = bind_arguments(api, fn, case.arguments)
    profile = api.analyze(fn, bound, fn.cache_key)
    assert before == (fn.src, tuple(fn.arg_names), fn.constexprs), "Original JIT metadata mutated"
    positions = [decision.position for decision in profile.decisions]
    assert sorted(positions) == sorted(fn.constexprs), "Missing or duplicate parameter classifications"
    assert set(profile.plan.dynamic) == {d.position for d in profile.decisions if d.classification.value == RUNTIME}
    assert bool(profile.recipe) == any(d.classification.value == SCHEDULE for d in profile.decisions)
    rows = []
    for decision in profile.decisions:
        name = fn.arg_names[decision.position]
        expected = case.expected[name]
        reasons = tuple(code for code, _ in decision.reasons)
        actual = decision.classification.value
        match = actual == expected.classification and (not expected.reason or expected.reason in reasons)
        gap = case.gaps.get(name)
        known = bool(not match and gap and actual == gap.observed and gap.reason in reasons)
        rows.append({"parameter": name, "expected": expected.classification, "actual": actual,
                     "why": expected.why, "required_reason": expected.reason,
                     "reasons": [{"code": code, "jit_relative_line": line} for code, line in decision.reasons],
                     "matches": match, "known_gap": gap.issue if known else None})
    return {"id": case.id, "collection": case.collection, "kind": case.kind, "kernel": case.kernel,
            "path": f"{case.collection}/src/kernels/{case.path}", "definition_line": fn.starting_line_number,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "jit_sha256": fn.cache_key,
            "arguments": case.arguments, "dynamic": [fn.arg_names[i] for i in profile.plan.dynamic],
            "recipe": asdict(profile.recipe) if profile.recipe else None, "parameters": rows}


def repository_info(path):
    def git(*args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"root": str(path), "revision": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def collect(triton_root, kernel_root):
    with analyzer(triton_root) as api:
        records = [analyze_case(api, kernel_root, case) for case in CASES]
        rule_version = api.config.RULE_VERSION
        # Hash the Python implementation actually imported, including local edits.
        root = Path(triton_root) / "third_party/ascend/backend/reuse"
        hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(root.rglob("*.py"))}
    rows = [row for record in records for row in record["parameters"]]
    return {"schema": 1, "mode": "source-analysis-only", "python": platform.python_version(),
            "rule_version": rule_version, "analyzer": repository_info(triton_root), "analyzer_sha256": hashes,
            "language_sha256": {name: hashlib.sha256((Path(triton_root) / "python/triton/language" / name).read_bytes()).hexdigest()
                                for name in ("__init__.py", "core.py", "math.py", "standard.py", "random.py")},
            "harness_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("analysis_adapter.py", "cases.py", "run_analysis.py")},
            "sources": {name: repository_info(Path(kernel_root) / name) for name in ("Q2TritonKernel", "Q3TritonKernel")},
            "summary": {"profiles": len(records), "definitions": len({(r['path'], r['kernel']) for r in records}),
                        "parameters": len(rows), "matching": sum(row["matches"] for row in rows),
                        "mismatching": sum(not row["matches"] for row in rows),
                        "known_gaps": dict(Counter(row["known_gap"] for row in rows if row["known_gap"])),
                        "expected_classes": dict(Counter(row["expected"] for row in rows)),
                        "actual_classes": dict(Counter(row["actual"] for row in rows))},
            "cases": records}


def markdown(report):
    summary = report["summary"]
    lines = ["# Q2/Q3 constexpr 纯分析验证", "",
             f"{summary['definitions']} 个 JIT 定义，{summary['profiles']} 个输入 profile，"
             f"{summary['parameters']} 项参数断言：{summary['matching']} 符合预期，{summary['mismatching']} 不符。", "",
             "使用当前 checkout 的分析器；没有执行 kernel、编译产物或验证 NPU 数值/ABI。",
             f"分析器版本：`{report['analyzer']['revision']}`，规则版本 {report['rule_version']}。",
             "源码及实现 SHA-256、工作区改动、完整绑定和分类理由见同名 JSON。", "",
             "`G1`：R3 形状用途不能独立证明 StaticRequired；缺少调度证明应为 Unknown。",
             "`G2`：浮点 pointee 的地址计算属于整数/指针运算，不能按浮点数据算术拒绝。", "",
             "## 分类差异", "", "| Profile | 参数 | 预期 | 实际 | 差异 |", "| --- | --- | --- | --- | --- |"]
    for record in report["cases"]:
        for row in record["parameters"]:
            if not row["matches"]:
                lines.append(f"| {record['id']} | {row['parameter']} | {row['expected']} | {row['actual']} | {row['known_gap'] or 'NEW'} |")
    lines.extend(["", "## 全部用例", "", "| Profile | 来源（定义行） | 符合/总数 |", "| --- | --- | --- |"])
    for record in report["cases"]:
        matches = sum(row["matches"] for row in record["parameters"])
        lines.append(f"| {record['id']} | {record['path']}:{record['definition_line']} `{record['kernel']}` | {matches}/{len(record['parameters'])} |")
    lines.extend(["", "理由行号为分析器原始 JIT 相对行号；helper 理由可能相对 helper，不能直接加到入口定义行。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-ascend-root", type=Path)
    parser.add_argument("--kernel-root", type=Path, help="TritonAscendTest root containing Q2TritonKernel and Q3TritonKernel")
    parser.add_argument("--output", type=Path, default=Path("reports/host_analysis"), help="Output stem for .json and .md")
    args = parser.parse_args(argv)
    try:
        triton_root = find_root(args.triton_ascend_root, "TRITON_ASCEND_ROOT", "triton-ascend")
        kernel_root = find_root(args.kernel_root, "TRITON_KERNEL_ROOT", "TritonAscendTest")
        report = collect(triton_root, kernel_root)
    except (FileNotFoundError, ValueError, AssertionError) as error:
        parser.exit(2, f"Analysis setup failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(report["summary"]["mismatching"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
