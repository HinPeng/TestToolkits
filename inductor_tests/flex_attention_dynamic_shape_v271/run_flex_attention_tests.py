#!/usr/bin/env python3
"""Run the root-directory FlexAttention envelope tests by letter.

Examples::

    python run_flex_attention_tests.py --list
    python run_flex_attention_tests.py a
    python run_flex_attention_tests.py --cases a,c-f
    python run_flex_attention_tests.py a-s --timeout 3600

Each selected letter is executed in a fresh pytest process.  A timestamped
run directory contains one log and the final ``torch_compile_debug/`` tree per
letter, plus JSON and Markdown summaries.  Inductor/Triton intermediate caches
are created outside the run directory and removed after the case finishes.
"""

from __future__ import annotations

import argparse
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
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "flex_attention_test_runs"
LETTER_MIN = "a"
LETTER_MAX = "s"
LETTERS = tuple(chr(code) for code in range(ord(LETTER_MIN), ord(LETTER_MAX) + 1))
CASE_FILE_RE = re.compile(r"^test_flex_attention_([a-s])_(.+)\.py$")


def discover_cases(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Discover available ``a``-through-``s`` test modules.

    The repository intentionally does not have a file for every letter.  The
    returned mapping therefore describes files that really exist instead of
    manufacturing empty cases for missing letters.
    """

    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("test_flex_attention_*.py")):
        match = CASE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        letter, suffix = match.groups()
        if letter in cases:
            raise ValueError(f"multiple test files found for case {letter!r}")
        cases[letter] = {
            "letter": letter,
            "name": f"{letter}_{suffix}",
            "file": path,
            "relative_file": str(path.relative_to(root)),
        }
    return dict(sorted(cases.items()))


def _split_selection(expression: str) -> list[str]:
    tokens = [token for token in re.split(r"[+,;\s]+", expression.strip()) if token]
    return tokens


def parse_selection(
    expression: str | None,
    available: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a selection such as ``a``, ``a,c-f`` or ``a-s``.

    A range may include letters for which no test file exists.  Those letters
    are returned separately so ``a-s`` remains useful with a sparse set of
    category files.  A directly requested missing letter is an error because
    it is almost certainly a typo.
    """

    if expression is None or not expression.strip() or expression.strip().lower() == "all":
        return list(available.values()), []

    selected_letters: list[str] = []
    missing_from_ranges: list[str] = []
    for token in _split_selection(expression.lower()):
        range_match = re.fullmatch(r"([a-s])-([a-s])", token)
        if range_match:
            start, end = range_match.groups()
            if start > end:
                raise ValueError(f"invalid descending case range: {token!r}")
            letters = LETTERS[ord(start) - ord(LETTER_MIN) : ord(end) - ord(LETTER_MIN) + 1]
            for letter in letters:
                if letter in available:
                    selected_letters.append(letter)
                else:
                    missing_from_ranges.append(letter)
            continue

        if not re.fullmatch(r"[a-s]+", token):
            raise ValueError(
                f"invalid case selection {token!r}; use letters a-s, for example a,c-f"
            )

        # Also accept compact selections such as ``acm``.  A compact token is
        # treated as explicit selection, so a missing letter is reported.
        for letter in token:
            if letter not in available:
                raise ValueError(
                    f"no test file found for case {letter!r}; use --list to see available cases"
                )
            selected_letters.append(letter)

    selected_letters = list(dict.fromkeys(selected_letters))
    if not selected_letters:
        raise ValueError("the selection does not contain an available test case")
    return [available[letter] for letter in selected_letters], list(dict.fromkeys(missing_from_ranges))


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    revision = result.stdout.strip()
    return revision or None


def _metadata(
    *,
    run_id: str,
    expression: str | None,
    selected: list[dict[str, Any]],
    missing_letters: list[str],
    argv: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_at": _timestamp(),
        "selection": expression or "all",
        "selected_letters": [case["letter"] for case in selected],
        "selected_files": [case["relative_file"] for case in selected],
        "missing_letters_in_ranges": missing_letters,
        "root": str(ROOT),
        "cwd": str(Path.cwd()),
        "command_line": [sys.executable, *argv],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "git_revision": _git_revision(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_HOME_PATH",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "TORCH_HOME",
                "CUDA_VISIBLE_DEVICES",
                "ASCEND_VISIBLE_DEVICES",
            )
            if os.environ.get(key)
        },
    }
    try:
        import importlib.metadata

        for distribution in ("torch", "torch_npu", "pytest"):
            try:
                metadata[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                metadata[distribution] = None
    except Exception as exc:  # pragma: no cover - defensive metadata only
        metadata["version_probe_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _pytest_summary(log_text: str) -> dict[str, int]:
    """Extract the final pytest result counts without depending on plugins."""

    counts = {key: 0 for key in ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")}
    summary_line = ""
    for line in reversed(log_text.splitlines()):
        if re.search(r"\d+\s+(?:passed|failed|skipped|error|errors|xfailed|xpassed)", line):
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
    """Describe final debug files without copying them outside the debug tree."""

    debug_dir = case_dir / "torch_compile_debug"
    if not debug_dir.is_dir():
        return {
            "directory": None,
            "file_count": 0,
            "fx_graph_files": [],
            "output_code_files": [],
        }

    files = sorted(path for path in debug_dir.rglob("*") if path.is_file())
    relative_files = [str(path.relative_to(case_dir)) for path in files]
    return {
        "directory": "torch_compile_debug",
        "file_count": len(relative_files),
        "fx_graph_files": [
            path for path in relative_files if Path(path).name.startswith("fx_graph")
        ],
        "output_code_files": [
            path for path in relative_files if Path(path).name == "output_code.py"
        ],
    }


def _run_one(
    case: dict[str, Any],
    run_dir: Path,
    timeout_seconds: int,
    emit: Any,
) -> dict[str, Any]:
    letter = str(case["letter"])
    case_dir = run_dir / letter
    case_dir.mkdir()
    log_path = case_dir / "run.log"
    command = [sys.executable, "-m", "pytest", "-v", "-s", str(case["file"])]
    command_text = shlex.join(command)
    temporary_cache_dir = Path(tempfile.mkdtemp(prefix=f"flex_attn_{letter}_cache_"))
    env = os.environ.copy()
    python_path = [str(ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "FLEX_ATTN_RUN_DIR": str(run_dir),
            "FLEX_ATTN_CASE_DIR": str(case_dir),
            "TORCH_COMPILE_DEBUG": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(temporary_cache_dir / "torchinductor"),
            "TRITON_CACHE_DIR": str(temporary_cache_dir / "triton"),
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONUNBUFFERED": "1",
        }
    )
    started_at = _now()
    monotonic_start = time.monotonic()
    timed_out = False
    returncode = 1

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"STARTED_AT={_timestamp(started_at)}\n")
        log.write(f"CASE={letter}\n")
        log.write(f"TEST_FILE={case['relative_file']}\n")
        log.write(f"COMMAND={command_text}\n")
        log.write(f"WORKING_DIRECTORY={case_dir}\n")
        log.write(f"TIMEOUT_SECONDS={timeout_seconds}\n\n")
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
                    f"\nCASE_TIMEOUT_SECONDS exceeded: {timeout_seconds}; process terminated.\n"
                )
        except OSError as exc:
            log.write(f"\nRUNNER_ERROR={type(exc).__name__}: {exc}\n")
            returncode = 125
        log.flush()

    # Keep only the final torch_compile_debug tree under the case directory.
    # The large Inductor/Triton intermediate caches live in the system temp
    # directory and are removed after this case finishes.
    shutil.rmtree(temporary_cache_dir, ignore_errors=True)

    elapsed_seconds = round(time.monotonic() - monotonic_start, 2)
    finished_at = _now()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    summary = _pytest_summary(log_text)
    debug_artifacts = _debug_artifacts(case_dir)
    status = _status(returncode, timed_out, summary)
    record: dict[str, Any] = {
        "letter": letter,
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
        "artifacts": [
            name
            for name in ("torch_compile_debug",)
            if (case_dir / name).exists()
        ],
        "summary": summary,
        "failure_tail": _failure_tail(log_path) if status in {"FAIL", "TIMEOUT"} else "",
    }
    emit(
        f"[{status}] {letter}: {case['relative_file']} "
        f"({elapsed_seconds}s, passed={summary['passed']}, failed={summary['failed']}, "
        f"skipped={summary['skipped']})"
    )
    return record


def _format_case_list(cases: dict[str, dict[str, Any]]) -> str:
    if not cases:
        return "(no a-s test files found)"
    return "\n".join(f"{letter}\t{case['relative_file']}" for letter, case in cases.items())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "selection",
        nargs="?",
        metavar="CASES",
        help="letters/ranges, for example a, a,c-f, or a-s; default: all available cases",
    )
    parser.add_argument(
        "--cases",
        "--case",
        "-c",
        dest="case_option",
        help="same as the positional CASES argument",
    )
    parser.add_argument("--list", action="store_true", help="list available a-s test files and exit")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT, help="root for timestamped run directories")
    parser.add_argument("--run-id", help="explicit run directory name; default: YYYYmmdd_HHMMSS")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="timeout per letter in seconds (default: 3600)",
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="stop after the first FAIL/TIMEOUT")
    parser.add_argument("--no-report", action="store_true", help="do not generate REPORT.md/REPORT.json")
    parser.add_argument("--dry-run", action="store_true", help="show selection and commands without running")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    available = discover_cases()
    if args.list:
        print(_format_case_list(available))
        missing = [letter for letter in LETTERS if letter not in available]
        if missing:
            print(f"missing\t{','.join(missing)}")
        return 0

    if args.selection and args.case_option:
        print("ERROR: provide CASES either positionally or with --cases, not both", file=sys.stderr)
        return 2
    expression = args.case_option or args.selection
    try:
        selected, missing_letters = parse_selection(expression, available)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("ERROR: --timeout must be positive", file=sys.stderr)
        return 2

    print(f"Selected cases: {','.join(case['letter'] for case in selected)}")
    if missing_letters:
        print(f"Range letters without files (skipped): {','.join(missing_letters)}")
    for case in selected:
        print(f"  {case['letter']}: {case['relative_file']}")
    if args.dry_run:
        for case in selected:
            command = [sys.executable, "-m", "pytest", "-v", "-s", str(case["file"])]
            print(f"DRY_RUN[{case['letter']}]: {shlex.join(command)}")
        return 0

    try:
        run_dir, run_id = _new_run_dir(args.output_root.expanduser().resolve(), args.run_id)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    metadata = _metadata(
        run_id=run_id,
        expression=expression,
        selected=selected,
        missing_letters=missing_letters,
        argv=sys.argv[1:] if argv is None else argv,
    )
    _write_json(run_dir / "run_metadata.json", metadata)
    runner_log_path = run_dir / "runner.log"
    records: list[dict[str, Any]] = []

    with runner_log_path.open("w", encoding="utf-8") as runner_log:
        def emit(message: str) -> None:
            line = f"[{_timestamp()}] {message}"
            print(line, flush=True)
            runner_log.write(line + "\n")
            runner_log.flush()

        emit(f"RUN_DIR={run_dir}")
        emit(f"SELECTION={expression or 'all'}")
        if missing_letters:
            emit(f"MISSING_RANGE_LETTERS={','.join(missing_letters)}")
        try:
            for case in selected:
                record = _run_one(case, run_dir, args.timeout, emit)
                records.append(record)
                _write_json(run_dir / "results.json", records)
                if args.stop_on_failure and record["status"] in {"FAIL", "TIMEOUT", "NO_TESTS"}:
                    emit(f"STOP_ON_FAILURE after case {case['letter']}")
                    break
        except KeyboardInterrupt:
            emit("INTERRUPTED_BY_USER")
            _write_json(run_dir / "results.json", records)
            return_code = 130
        else:
            return_code = 0

    metadata["finished_at"] = _timestamp()
    metadata["completed_cases"] = [record["letter"] for record in records]
    metadata["status"] = "INTERRUPTED" if return_code == 130 else "COMPLETED"
    _write_json(run_dir / "run_metadata.json", metadata)
    _write_json(run_dir / "results.json", records)

    report_status = None
    report_path = run_dir / "REPORT.md"
    if not args.no_report:
        try:
            from dump_flex_attention_report import build_report

            markdown, report_data = build_report(run_dir, records)
            report_path.write_text(markdown, encoding="utf-8")
            _write_json(run_dir / "REPORT.json", report_data)
            report_status = str(report_data["overall_status"])
            print(f"REPORT={report_path}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: report generation failed: {exc}", file=sys.stderr)
            return_code = return_code or 2

    failed = [record for record in records if record["status"] in {"FAIL", "TIMEOUT", "NO_TESTS"}]
    if failed:
        return_code = return_code or 1
    print(f"RUN_DIR={run_dir}")
    if report_status:
        print(f"STATUS={report_status}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
