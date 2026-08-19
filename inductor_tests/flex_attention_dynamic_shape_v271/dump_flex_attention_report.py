#!/usr/bin/env python3
"""Render a durable Markdown/JSON report from a FlexAttention test run.

The runner stores raw logs and ``results.json``.  This command is intentionally
standalone so a report can be regenerated on a server after a timeout, or on a
workstation after copying only the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "flex_attention_test_runs"


def _latest_run() -> Path:
    candidates = [path for path in RUN_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run directories under {RUN_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_case_json(log_text: str) -> dict[str, Any] | None:
    for line in reversed(log_text.splitlines()):
        if line.startswith("CASE_RESULT_JSON="):
            try:
                return json.loads(line.removeprefix("CASE_RESULT_JSON="))
            except json.JSONDecodeError:
                return None
    return None


def _tail(path: Path, lines: int = 25) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    result_file = run_dir / "results.json"
    if result_file.is_file():
        data = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected a list in {result_file}")
        return data

    # Fallback for a manually collected run directory from an older runner.
    records: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        log_path = case_dir / "run.log"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        details = _parse_case_json(text)
        timed_out = "CASE_TIMEOUT_SECONDS exceeded" in text
        if timed_out:
            status = "TIMEOUT"
        elif details and details.get("status") == "PASS":
            status = "PASS"
        elif "OK" in text and "FAILED" not in text:
            status = "PASS"
        else:
            status = "FAIL"
        records.append(
            {
                "case": case_dir.name,
                "status": status,
                "run_log": str(log_path.relative_to(run_dir)) if log_path.is_file() else "",
                "details": details or {},
                "failure_tail": _tail(log_path),
            }
        )
    return records


def _md(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _details(record: dict[str, Any]) -> dict[str, Any]:
    details = record.get("details")
    if isinstance(details, dict):
        return details
    # The first runner version flattened the custom case JSON into the record.
    ignored = {
        "case",
        "suite",
        "status",
        "returncode",
        "elapsed_seconds",
        "run_log",
        "output_code_files",
        "timed_out",
        "command",
        "selector",
        "failure_tail",
    }
    return {key: value for key, value in record.items() if key not in ignored}


def _observation(record: dict[str, Any]) -> str:
    details = _details(record)
    for key in ("observations", "stats", "error", "stream_mode", "checks"):
        if key in details:
            return _md(details[key])
    if record.get("status") == "SKIP":
        return "unittest skipped"
    return "-"


def _status_counts(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(record.get("status", "FAIL")) for record in records)


def _overall_status(records: list[dict[str, Any]]) -> str:
    statuses = {str(record.get("status", "FAIL")) for record in records}
    if "FAIL" in statuses or "TIMEOUT" in statuses:
        return "FAIL"
    if "SKIP" in statuses:
        return "PASS_WITH_SKIP"
    return "PASS"


def build_report(run_dir: Path, records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    metadata_path = run_dir / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata = loaded

    counts = _status_counts(records)
    overall = _overall_status(records)
    generated_at = datetime.now(timezone.utc).isoformat()
    report_data = {
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "overall_status": overall,
        "counts": dict(counts),
        "metadata": metadata,
        "records": records,
    }

    lines = [
        "# Dynamic Shape FlexAttention v2.7.1 测试报告",
        "",
        f"- 总结状态：`{overall}`",
        f"- 测试批次：`{run_dir.name}`",
        f"- 运行目录：`{run_dir}`",
        f"- 生成时间（UTC）：`{generated_at}`",
    ]
    for key in ("hostname", "python", "torch", "torch_npu", "npu_available"):
        if key in metadata:
            lines.append(f"- {key}：`{_md(metadata[key])}`")
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            "| 用例 | 套件 | 状态 | 观察/错误 | 用时(s) | 日志 | debug code |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for record in records:
        output_code = record.get("output_code_files", [])
        output_code_text = ", ".join(f"`{item}`" for item in output_code) or "-"
        log = record.get("run_log", "-")
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(record.get("case")),
                    _md(record.get("suite", "-")),
                    _md(record.get("status", "FAIL")),
                    _observation(record),
                    _md(record.get("elapsed_seconds", "-")),
                    f"`{log}`",
                    output_code_text,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 统计",
            "",
            "、".join(f"`{key}`={value}" for key, value in sorted(counts.items())) or "无记录",
            "",
            "## 用例说明",
            "",
            "- `S*`：本地动态 shape exact-capacity 主验收；覆盖 `T=0/1/C`、动态 Q/KV、尾块、非连续 metadata、backward、stream 和 codegen 顺序。",
            "- `P*`：torch_npu patch 随附的源码/runtime 契约用例。",
            "- `M*`：TritonAutomation NPU 社区文件中筛选出的动态 shape 用例；覆盖显式/自动动态化、动态 batch/free symbol、kernel options、max-autotune、stride 和非连续布局。",
            "- `output_code.py` 与完整 `torch_compile_debug/` 保留在对应用例目录，失败/超时也不清理。",
            "",
            "## 失败详情",
            "",
        ]
    )
    failures = [record for record in records if record.get("status") in {"FAIL", "TIMEOUT"}]
    if not failures:
        lines.append("没有 FAIL/TIMEOUT 用例。")
    for record in failures:
        lines.extend(
            [
                f"### `{record.get('case', '?')}`",
                "",
                f"- 命令：`{_md(' '.join(record.get('command', [])))}`",
                f"- 错误：`{_md(_details(record).get('error', record.get('status', 'FAIL')))}`",
                "",
                "```text",
                str(record.get("failure_tail", "")).rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n", report_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="test run directory; default is the newest directory under flex_attention_test_runs",
    )
    parser.add_argument("--output", type=Path, help="Markdown output path")
    parser.add_argument("--json-output", type=Path, help="normalized JSON output path")
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="return non-zero when one or more unittest cases are skipped",
    )
    args = parser.parse_args()

    run_dir = (args.run_dir or _latest_run()).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2
    try:
        records = _load_records(run_dir)
        markdown, report_data = build_report(run_dir, records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot build report: {exc}", file=sys.stderr)
        return 2

    output = (args.output or run_dir / "REPORT.md").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_output = args.json_output or run_dir / "REPORT.json"
    json_output = json_output.expanduser().resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"STATUS={report_data['overall_status']}")
    print(f"REPORT={output}")
    print(f"REPORT_JSON={json_output}")
    if report_data["overall_status"] == "FAIL":
        return 1
    if args.fail_on_skip and report_data["overall_status"] == "PASS_WITH_SKIP":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
