#!/usr/bin/env python3
"""Generate Markdown and JSON reports for a FlexAttention test run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "flex_attention_test_runs"


def _latest_run() -> Path:
    if not RUN_ROOT.is_dir():
        raise FileNotFoundError(f"run root does not exist: {RUN_ROOT}")
    candidates = [path for path in RUN_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run directories under {RUN_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    loaded = _load_json(run_dir / "results.json", None)
    if loaded is not None:
        if not isinstance(loaded, list):
            raise ValueError(f"expected a list in {run_dir / 'results.json'}")
        return [record for record in loaded if isinstance(record, dict)]

    # Fallback for a partially copied run directory.  The normal runner always
    # writes results.json incrementally, but this keeps report generation useful
    # after an abrupt interruption.
    records: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        log_path = case_dir / "run.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if "CASE_TIMEOUT_SECONDS exceeded" in log_text:
            status = "TIMEOUT"
        elif " failed" in log_text or "ERROR" in log_text:
            status = "FAIL"
        elif " passed" in log_text:
            status = "PASS"
        elif " skipped" in log_text:
            status = "SKIP"
        else:
            status = "NO_TESTS"
        records.append(
            {
                "letter": case_dir.name,
                "case": case_dir.name,
                "status": status,
                "run_log": str(log_path.relative_to(run_dir)) if log_path.is_file() else "",
                "elapsed_seconds": None,
                "summary": {},
                "failure_tail": _tail(log_path) if status in {"FAIL", "TIMEOUT"} else "",
            }
        )
    return records


def _md(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", r"\|").replace("\n", " ")


def _relative_link(run_dir: Path, relative_path: str) -> str:
    if not relative_path:
        return "-"
    path = Path(relative_path)
    if path.is_absolute():
        try:
            path = path.relative_to(run_dir)
        except ValueError:
            return _md(relative_path)
    return f"[{path.as_posix()}]({path.as_posix()})"


def _summary(record: dict[str, Any]) -> str:
    summary = record.get("summary")
    if not isinstance(summary, dict) or not summary:
        return "-"
    keys = ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
    values = [f"{key}={summary[key]}" for key in keys if summary.get(key)]
    return ", ".join(values) or "0 tests"


def _status_counts(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(record.get("status", "FAIL")) for record in records)


def _overall_status(records: list[dict[str, Any]]) -> str:
    if not records:
        return "FAIL"
    statuses = {str(record.get("status", "FAIL")) for record in records}
    if statuses & {"FAIL", "TIMEOUT", "NO_TESTS"}:
        return "FAIL"
    if statuses & {"SKIP", "PASS_WITH_SKIP"}:
        return "PASS_WITH_SKIP"
    return "PASS"


def build_report(run_dir: Path, records: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    metadata = _load_json(run_dir / "run_metadata.json", {})
    if not isinstance(metadata, dict):
        metadata = {}
    counts = _status_counts(records)
    overall_status = _overall_status(records)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report_data: dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "overall_status": overall_status,
        "counts": dict(counts),
        "metadata": metadata,
        "records": records,
    }

    lines = [
        "# FlexAttention 动态 Shape 测试报告",
        "",
        f"- 总体状态：`{overall_status}`",
        f"- 测试批次：`{run_dir.name}`",
        f"- 生成时间：`{generated_at}`",
    ]
    for key, label in (
        ("selection", "选择"),
        ("started_at", "开始时间"),
        ("finished_at", "结束时间"),
        ("hostname", "主机"),
        ("python", "Python"),
        ("torch", "torch"),
        ("torch_npu", "torch_npu"),
        ("git_revision", "Git"),
    ):
        if key in metadata and metadata[key] is not None:
            lines.append(f"- {label}：`{_md(metadata[key])}`")
    missing = metadata.get("missing_letters_in_ranges")
    if missing:
        lines.append(f"- 范围内无文件的字母：`{_md(missing)}`")

    lines.extend(
        [
            "",
            "## 汇总",
            "",
            "| 字母 | 测试文件 | 状态 | 用时(s) | pytest 结果 | 日志 |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(record.get("letter", "?")),
                    _md(record.get("test_file", record.get("case", "-"))),
                    _md(record.get("status", "FAIL")),
                    _md(record.get("elapsed_seconds")),
                    _md(_summary(record)),
                    _relative_link(run_dir, str(record.get("run_log", ""))),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 统计",
            "",
            ", ".join(f"`{key}`={value}" for key, value in sorted(counts.items())) or "无记录",
            "",
            "## 产物",
            "",
            "每个字母的目录中只保留 `run.log` 和最终的 `torch_compile_debug/`；"
            "其中包含 `fx_graph*`、`output_code.py` 等调试产物。Inductor/Triton "
            "中间 cache 位于系统临时目录，测试结束后会清理。",
            "",
            "## 失败详情",
            "",
        ]
    )
    failures = [
        record
        for record in records
        if record.get("status") in {"FAIL", "TIMEOUT", "NO_TESTS"}
    ]
    if not failures:
        lines.append("没有 FAIL/TIMEOUT/NO_TESTS 用例。")
    for record in failures:
        tail = str(record.get("failure_tail", "")).rstrip()
        if not tail and record.get("run_log"):
            tail = _tail(run_dir / str(record["run_log"]))
        lines.extend(
            [
                f"### `{_md(record.get('letter', record.get('case', '?')))}`",
                "",
                f"- 状态：`{_md(record.get('status'))}`",
                f"- 命令：`{_md(' '.join(record.get('command', [])))}`",
                "",
                "```text",
                tail,
                "```",
                "",
            ]
        )

    return "\n".join(lines) + "\n", report_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="run directory; default: newest directory under flex_attention_test_runs",
    )
    parser.add_argument("--output", type=Path, help="Markdown output path; default: RUN_DIR/REPORT.md")
    parser.add_argument("--json-output", type=Path, help="JSON output path; default: RUN_DIR/REPORT.json")
    parser.add_argument("--fail-on-skip", action="store_true", help="return 1 for skipped cases too")
    args = parser.parse_args(argv)

    try:
        run_dir = (args.run_dir or _latest_run()).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        records = _load_records(run_dir)
        markdown, report_data = build_report(run_dir, records)
        output = (args.output or run_dir / "REPORT.md").expanduser().resolve()
        json_output = (args.json_output or run_dir / "REPORT.json").expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        json_output.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot build report: {exc}", file=sys.stderr)
        return 2

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
