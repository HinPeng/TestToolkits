#!/usr/bin/env python3
"""Batch-run the score_mod dynamic-shape tests and preserve diagnostics.

Examples::

    python run_score_mod_tests.py --list
    python run_score_mod_tests.py a
    python run_score_mod_tests.py --cases a,c-d
    python run_score_mod_tests.py score_mod_a
    python run_score_mod_tests.py all --timeout 14400

Each selected category runs in a fresh pytest process. A timestamped run
directory stores the complete log and final ``torch_compile_debug/`` tree for
every category, together with JSON and Markdown summaries. Inductor and Triton
intermediate caches use a system temporary directory and are removed after the
category finishes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "score_mod_test_runs"
CASE_FILE_RE = re.compile(r"^test_score_mod_([a-d])_(.+)\.py$")


def discover_cases(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Discover the score_mod A-D modules that actually exist."""

    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("test_score_mod_[a-d]_*.py")):
        match = CASE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        case_id, suffix = match.groups()
        if case_id in cases:
            raise ValueError(f"multiple test files found for case {case_id!r}")
        cases[case_id] = {
            "id": case_id,
            "name": f"score_mod_{case_id}_{suffix.removesuffix('_dynamic')}",
            "file": path,
            "relative_file": str(path.relative_to(root)),
        }
    return dict(sorted(cases.items()))


def _split_selection(expression: str) -> list[str]:
    return [
        token
        for token in re.split(r"[+,;\s]+", expression.strip().lower())
        if token
    ]


def _normalize_case_id(token: str) -> str:
    alias = re.fullmatch(r"score_mod_([a-d])", token)
    return alias.group(1) if alias else token


def parse_selection(
    expression: str | None,
    available: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse selections such as ``a``, ``a,c-d``, ``abcd`` or ``all``."""

    if (
        expression is None
        or not expression.strip()
        or expression.strip().lower() == "all"
    ):
        return list(available.values())

    selected_ids: list[str] = []
    for raw_token in _split_selection(expression):
        token = _normalize_case_id(raw_token)
        range_match = re.fullmatch(r"([a-d])-([a-d])", token)
        if range_match:
            start, end = range_match.groups()
            if start > end:
                raise ValueError(f"invalid descending case range: {raw_token!r}")
            selected_ids.extend(
                chr(code) for code in range(ord(start), ord(end) + 1)
            )
            continue

        if not re.fullmatch(r"[a-d]+", token):
            raise ValueError(
                f"invalid case selection {raw_token!r}; use a-d or score_mod_a"
            )
        selected_ids.extend(token)

    selected_ids = list(dict.fromkeys(selected_ids))
    missing = [case_id for case_id in selected_ids if case_id not in available]
    if missing:
        raise ValueError(
            f"no test file found for case(s) {','.join(missing)}; use --list"
        )
    if not selected_ids:
        raise ValueError("the selection does not contain an available test case")
    return [available[case_id] for case_id in selected_ids]


def _now() -> datetime:
    return datetime.now().astimezone()


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _new_run_dir(output_root: Path, requested_id: str | None) -> tuple[Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    if requested_id:
        if requested_id in {".", ".."} or Path(requested_id).name != requested_id:
            raise ValueError("--run-id must be a single directory name")
        run_dir = output_root / requested_id
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise ValueError(f"run directory already exists: {run_dir}") from exc
        return run_dir, requested_id

    base = _now().strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        run_id = base if index == 0 else f"{base}_{index:02d}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir, run_id
    raise ValueError(f"could not allocate a run directory under {output_root}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(
    *,
    run_id: str,
    expression: str | None,
    selected: list[dict[str, Any]],
    argv: list[str],
) -> dict[str, Any]:
    environment_keys = (
        "ASCEND_HOME_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "TORCH_HOME",
        "ASCEND_VISIBLE_DEVICES",
        "TORCH_LOGS",
    )
    environment = {
        key: os.environ[key] for key in environment_keys if os.environ.get(key)
    }
    environment.setdefault("TORCH_LOGS", "recompiles")
    return {
        "run_id": run_id,
        "started_at": _timestamp(),
        "selection": expression or "all",
        "selected_cases": [case["id"] for case in selected],
        "selected_files": [case["relative_file"] for case in selected],
        "root": str(ROOT),
        "cwd": str(Path.cwd()),
        "command_line": [sys.executable, *argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "git_revision": _git_revision(),
        "torch": _distribution_version("torch"),
        "torch_npu": _distribution_version("torch_npu"),
        "pytest": _distribution_version("pytest"),
        "environment": environment,
    }


def _pytest_summary(log_text: str) -> dict[str, int]:
    """Extract final pytest counts without depending on a pytest plugin."""

    counts = {
        key: 0
        for key in ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
    }
    summary_line = ""
    for line in reversed(log_text.splitlines()):
        if re.search(
            r"\d+\s+(?:passed|failed|skipped|error|errors|xfailed|xpassed)",
            line,
        ):
            summary_line = line
            break
    patterns = {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "skipped": r"(\d+)\s+skipped",
        "errors": r"(\d+)\s+errors?",
        "xfailed": r"(\d+)\s+xfailed",
        "xpassed": r"(\d+)\s+xpassed",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, summary_line)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _failure_tail(path: Path, line_count: int = 60) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows fallback
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _status(returncode: int, timed_out: bool, summary: dict[str, int]) -> str:
    if timed_out:
        return "TIMEOUT"
    if returncode == 5:
        return "NO_TESTS"
    if returncode != 0 or summary["failed"] or summary["errors"]:
        return "FAIL"
    if summary["skipped"] and not summary["passed"]:
        return "SKIP"
    if summary["skipped"]:
        return "PASS_WITH_SKIP"
    return "PASS"


def _debug_artifacts(case_dir: Path) -> dict[str, Any]:
    """Inventory the final torch_compile_debug tree retained for a case."""

    debug_dir = case_dir / "torch_compile_debug"
    if not debug_dir.is_dir():
        return {
            "directory": None,
            "file_count": 0,
            "total_bytes": 0,
            "fx_graph_files": [],
            "output_code_files": [],
        }

    files = sorted(path for path in debug_dir.rglob("*") if path.is_file())
    relative_files = [str(path.relative_to(case_dir)) for path in files]
    return {
        "directory": str(debug_dir.relative_to(case_dir)),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "fx_graph_files": [
            path for path in relative_files if Path(path).name.startswith("fx_graph")
        ],
        "output_code_files": [
            path for path in relative_files if Path(path).name == "output_code.py"
        ],
    }


def _pytest_command(case: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-s",
        "-ra",
        str(case["file"]),
    ]


def _run_one(
    case: dict[str, Any],
    run_dir: Path,
    timeout_seconds: int,
    emit: Callable[[str], None],
) -> dict[str, Any]:
    case_id = str(case["id"])
    case_dir = run_dir / case_id
    case_dir.mkdir()
    log_path = case_dir / "run.log"
    command = _pytest_command(case)
    command_text = shlex.join(command)
    temporary_cache_dir = Path(
        tempfile.mkdtemp(prefix=f"score_mod_{case_id}_cache_")
    )
    env = os.environ.copy()
    # The default debug root is relative to the subprocess working directory.
    # Drop inherited redirects so the retained tree is always case-local.
    env.pop("TORCH_COMPILE_DEBUG_DIR", None)
    env.pop("TORCH_LOGS_OUT", None)
    python_path = [str(ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "SCORE_MOD_RUN_DIR": str(run_dir),
            "SCORE_MOD_CASE_DIR": str(case_dir),
            "FLEX_ATTN_RUN_DIR": str(run_dir),
            "FLEX_ATTN_CASE_DIR": str(case_dir),
            "TORCH_COMPILE_DEBUG": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(temporary_cache_dir / "torchinductor"),
            "TRITON_CACHE_DIR": str(temporary_cache_dir / "triton"),
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.setdefault("TORCH_LOGS", "recompiles")
    started_at = _now()
    monotonic_start = time.monotonic()
    timed_out = False
    returncode = 1
    process: subprocess.Popen[Any] | None = None

    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"STARTED_AT={_timestamp(started_at)}\n")
            log.write(f"CASE={case_id}\n")
            log.write(f"TEST_FILE={case['relative_file']}\n")
            log.write(f"COMMAND={command_text}\n")
            log.write(f"WORKING_DIRECTORY={case_dir}\n")
            log.write(f"TIMEOUT_SECONDS={timeout_seconds}\n")
            log.write(f"TORCHINDUCTOR_CACHE_DIR={env['TORCHINDUCTOR_CACHE_DIR']}\n")
            log.write(f"TRITON_CACHE_DIR={env['TRITON_CACHE_DIR']}\n\n")
            log.flush()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=case_dir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name == "posix"),
                )
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process(process)
                    returncode = 124
                    log.write(
                        "\nCASE_TIMEOUT_SECONDS exceeded: "
                        f"{timeout_seconds}; process terminated.\n"
                    )
            except OSError as exc:
                log.write(f"\nRUNNER_ERROR={type(exc).__name__}: {exc}\n")
                returncode = 125
            except KeyboardInterrupt:
                if process is not None:
                    _terminate_process(process)
                log.write("\nINTERRUPTED_BY_USER; subprocess terminated.\n")
                raise
            finally:
                log.flush()
    finally:
        # Preserve torch_compile_debug in case_dir, but remove large temporary
        # Inductor/Triton caches just like the parent-directory runner.
        shutil.rmtree(temporary_cache_dir, ignore_errors=True)

    elapsed_seconds = round(time.monotonic() - monotonic_start, 2)
    finished_at = _now()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = _pytest_summary(log_text)
    debug_artifacts = _debug_artifacts(case_dir)
    status = _status(returncode, timed_out, summary)
    record: dict[str, Any] = {
        "id": case_id,
        "case": case["name"],
        "test_file": case["relative_file"],
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "started_at": _timestamp(started_at),
        "finished_at": _timestamp(finished_at),
        "elapsed_seconds": elapsed_seconds,
        "command": command,
        "run_log": str(log_path.relative_to(run_dir)),
        "debug_artifacts": debug_artifacts,
        "summary": summary,
        "failure_tail": (
            _failure_tail(log_path)
            if status in {"FAIL", "TIMEOUT", "NO_TESTS"}
            else ""
        ),
    }
    emit(
        f"[{status}] {case_id}: {case['relative_file']} "
        f"({elapsed_seconds}s, passed={summary['passed']}, "
        f"failed={summary['failed']}, skipped={summary['skipped']}, "
        f"debug_files={debug_artifacts['file_count']})"
    )
    return record


def _overall_status(records: list[dict[str, Any]]) -> str:
    if not records:
        return "FAIL"
    statuses = {str(record.get("status", "FAIL")) for record in records}
    if statuses & {"FAIL", "TIMEOUT", "NO_TESTS"}:
        return "FAIL"
    if statuses & {"SKIP", "PASS_WITH_SKIP"}:
        return "PASS_WITH_SKIP"
    return "PASS"


def _summary_text(summary: dict[str, Any]) -> str:
    keys = ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
    parts = [f"{key}={summary[key]}" for key in keys if summary.get(key)]
    return ", ".join(parts) or "0 tests"


def _md(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("|", r"\|").replace("\n", " ")


def _build_report(
    run_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    overall_status = _overall_status(records)
    generated_at = _timestamp()
    report_data = {
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "overall_status": overall_status,
        "metadata": metadata,
        "records": records,
    }
    lines = [
        "# score_mod 动态 Shape 测试报告",
        "",
        f"- 总体状态：`{overall_status}`",
        f"- 测试批次：`{run_dir.name}`",
        f"- 生成时间：`{generated_at}`",
        f"- 主机：`{_md(metadata.get('hostname'))}`",
        f"- torch：`{_md(metadata.get('torch'))}`",
        f"- torch_npu：`{_md(metadata.get('torch_npu'))}`",
        f"- Git：`{_md(metadata.get('git_revision'))}`",
        "",
        "## 汇总",
        "",
        "| Case | 测试文件 | 状态 | 用时(s) | pytest 结果 | 编译产物 | 日志 |",
        "|---|---|---|---:|---|---|---|",
    ]
    for record in records:
        debug = record.get("debug_artifacts", {})
        log_path = str(record.get("run_log", ""))
        log_link = f"[{log_path}]({log_path})" if log_path else "-"
        debug_directory = debug.get("directory")
        if debug_directory:
            debug_path = f"{record['id']}/{debug_directory}"
            debug_link = (
                f"[{debug.get('file_count', 0)} files]({debug_path}/)"
            )
        else:
            debug_link = "0"
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(record.get("id")),
                    _md(record.get("test_file")),
                    _md(record.get("status")),
                    _md(record.get("elapsed_seconds")),
                    _md(_summary_text(record.get("summary", {}))),
                    debug_link,
                    log_link,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 产物",
            "",
            "每个 case 目录保存完整 `run.log` 和最终 `torch_compile_debug/`；",
            "其中包含 `fx_graph*`、`output_code.py` 等调试产物。Inductor/Triton",
            "中间 cache 位于系统临时目录，并在对应 case 结束后清理。",
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
        lines.extend(
            [
                f"### `{_md(record.get('id'))}`",
                "",
                f"- 状态：`{_md(record.get('status'))}`",
                f"- 命令：`{_md(shlex.join(record.get('command', [])))}`",
                "",
                "```text",
                str(record.get("failure_tail", "")).rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n", report_data


def _format_case_list(cases: dict[str, dict[str, Any]]) -> str:
    if not cases:
        return "(no score_mod test cases found)"
    return "\n".join(
        f"{case_id}\tscore_mod_{case_id}\t{case['relative_file']}"
        for case_id, case in cases.items()
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "selection",
        nargs="?",
        metavar="CASES",
        help="case IDs/ranges, for example a, a,c-d, a-d or all (default: all)",
    )
    parser.add_argument(
        "--cases",
        "--case",
        "-c",
        dest="case_option",
        help="same as the positional CASES argument",
    )
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUN_ROOT,
        help="root for timestamped run directories",
    )
    parser.add_argument(
        "--run-id",
        help="explicit run directory name; default: YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="timeout per case in seconds (default: 14400)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop after the first FAIL/TIMEOUT/NO_TESTS",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="do not generate REPORT.md/REPORT.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show selected pytest commands without running",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(raw_argv)
    available = discover_cases()
    if args.list:
        print(_format_case_list(available))
        return 0

    if args.selection and args.case_option:
        print(
            "ERROR: provide CASES either positionally or with --cases, not both",
            file=sys.stderr,
        )
        return 2
    expression = args.case_option or args.selection
    try:
        selected = parse_selection(expression, available)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("ERROR: --timeout must be positive", file=sys.stderr)
        return 2

    print(f"Selected cases: {','.join(case['id'] for case in selected)}")
    for case in selected:
        print(f"  {case['id']}: {case['relative_file']}")
    if args.dry_run:
        for case in selected:
            print(
                f"DRY_RUN[{case['id']}]: {shlex.join(_pytest_command(case))}"
            )
        return 0

    try:
        run_dir, run_id = _new_run_dir(
            args.output_root.expanduser().resolve(),
            args.run_id,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    metadata = _metadata(
        run_id=run_id,
        expression=expression,
        selected=selected,
        argv=raw_argv,
    )
    _write_json(run_dir / "run_metadata.json", metadata)
    runner_log_path = run_dir / "runner.log"
    records: list[dict[str, Any]] = []
    return_code = 0

    with runner_log_path.open("w", encoding="utf-8") as runner_log:

        def emit(message: str) -> None:
            line = f"[{_timestamp()}] {message}"
            print(line, flush=True)
            runner_log.write(line + "\n")
            runner_log.flush()

        emit(f"RUN_DIR={run_dir}")
        emit(f"SELECTION={expression or 'all'}")
        try:
            for case in selected:
                record = _run_one(case, run_dir, args.timeout, emit)
                records.append(record)
                _write_json(run_dir / "results.json", records)
                if args.stop_on_failure and record["status"] in {
                    "FAIL",
                    "TIMEOUT",
                    "NO_TESTS",
                }:
                    emit(f"STOP_ON_FAILURE after case {case['id']}")
                    break
        except KeyboardInterrupt:
            emit("INTERRUPTED_BY_USER")
            return_code = 130

    metadata["finished_at"] = _timestamp()
    metadata["completed_cases"] = [record["id"] for record in records]
    metadata["status"] = "INTERRUPTED" if return_code == 130 else "COMPLETED"
    _write_json(run_dir / "run_metadata.json", metadata)
    _write_json(run_dir / "results.json", records)

    report_status: str | None = None
    if not args.no_report:
        markdown, report_data = _build_report(run_dir, metadata, records)
        (run_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
        _write_json(run_dir / "REPORT.json", report_data)
        report_status = str(report_data["overall_status"])
        print(f"REPORT={run_dir / 'REPORT.md'}")

    if any(
        record["status"] in {"FAIL", "TIMEOUT", "NO_TESTS"}
        for record in records
    ):
        return_code = return_code or 1
    print(f"RUN_DIR={run_dir}")
    if report_status:
        print(f"STATUS={report_status}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
