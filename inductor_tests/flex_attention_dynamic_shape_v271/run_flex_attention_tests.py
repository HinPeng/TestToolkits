#!/usr/bin/env python3
"""Run the v2.7.1 dynamic-shape FlexAttention suites on an NPU server.

Every case is launched in a fresh Python process and receives its own
``TORCH_COMPILE_DEBUG``/Inductor/Triton cache directory.  This makes a timeout
or compiler crash local to one case and leaves enough evidence for the report
dump script.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CUSTOM_TEST = ROOT / "test_dynamic_flex_attention.py"
COMMUNITY_TEST = ROOT / "test_flex_attention_dynamic_mask_out.py"
RUN_ROOT = ROOT / "flex_attention_test_runs"

CUSTOM_CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "S00_source_contracts",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 180,
    },
    {
        "case": "C01_capacity_reuse",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 300,
    },
    {
        "case": "C02_dynamic_shape_reuse",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 300,
    },
    {
        "case": "C03_non_aligned_modes",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 300,
    },
    {
        "case": "C04_non_contiguous_metadata",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 300,
    },
    {
        "case": "C05_backward_capacity_and_grad",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 600,
    },
    {
        "case": "C06_non_default_stream",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 300,
    },
    {
        "case": "C07_codegen_order",
        "suite": "dynamic",
        "kind": "custom",
        "timeout": 300,
    },
)

COMMUNITY_SOURCE_METHODS = (
    "test_eager_compact_metadata_is_removed",
    "test_compact_metadata_is_split_into_offsets_and_mapping",
    "test_compact_kernels_use_symbolic_mapping_sizes",
    "test_forward_compact_kernel_uses_symbolic_mapping_sizes",
    "test_backward_compact_kernels_use_symbolic_mapping_sizes",
    "test_lowering_allocates_symbolic_actual_capacity",
    "test_legacy_static_total_options_are_removed",
    "test_dynamic_backward_disables_static_tasklist_codegen",
    "test_symbolic_sequence_lengths_do_not_require_python_truth_values",
    "test_default_noop_block_mask_uses_mask_in_forward",
    "test_backward_dq_task_count_comes_from_runtime_q_shape",
    "test_backward_block_position_covers_all_runtime_kv_blocks",
)
COMMUNITY_NPU_METHODS = (
    "test_create_block_mask_has_no_compact_metadata_capture",
    "test_forward_reuses_graph_for_dynamic_q_kv_and_sparse_z",
    "test_backward_reuses_graph_for_dynamic_q_kv_and_sparse_z",
    "test_full_mask_uses_runtime_compact_metadata",
)


def _community_cases() -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for index, method in enumerate(COMMUNITY_SOURCE_METHODS, start=1):
        cases.append(
            {
                "case": f"M{index:02d}_{method.removeprefix('test_')}",
                "suite": "community",
                "kind": "community",
                "selector": f"TestFlexAttentionDynamicMaskOutSource.{method}",
                "timeout": 180,
            }
        )
    offset = len(COMMUNITY_SOURCE_METHODS) + 1
    for index, method in enumerate(COMMUNITY_NPU_METHODS, start=offset):
        cases.append(
            {
                "case": f"M{index:02d}_{method.removeprefix('test_')}",
                "suite": "community",
                "kind": "community",
                "selector": f"TestFlexAttentionDynamicMaskOutNPU.{method}",
                "timeout": 600 if "backward" in method else 300,
            }
        )
    return tuple(cases)


COMMUNITY_CASES = _community_cases()
ALL_CASES = CUSTOM_CASES + COMMUNITY_CASES
CASE_BY_NAME = {case["case"]: case for case in ALL_CASES}


def _parse_case_result(log_text: str) -> dict[str, Any] | None:
    for line in reversed(log_text.splitlines()):
        if line.startswith("CASE_RESULT_JSON="):
            try:
                return json.loads(line.removeprefix("CASE_RESULT_JSON="))
            except json.JSONDecodeError:
                return None
    return None


def _copy_debug_files(case_dir: Path) -> list[str]:
    sources = sorted(case_dir.glob("torch_compile_debug/**/output_code.py"))
    copied: list[str] = []
    for index, source in enumerate(sources):
        name = "output_code.py" if index == 0 else f"output_code_{index}.py"
        destination = case_dir / name
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(case_dir)))
    return copied


def _failure_tail(log_path: Path, line_count: int = 40) -> str:
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _command_for(case: dict[str, Any]) -> list[str]:
    if case["kind"] == "custom":
        return [sys.executable, str(CUSTOM_TEST), "--case", case["case"]]
    return [sys.executable, str(COMMUNITY_TEST), str(case["selector"])]


def _metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "script_root": str(ROOT),
        "npu_available": "unknown_until_case_import",
    }
    try:
        import importlib.metadata

        metadata["torch"] = importlib.metadata.version("torch")
        metadata["torch_npu"] = importlib.metadata.version("torch_npu")
    except Exception as exc:
        metadata["version_probe_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _status_for(record: dict[str, Any], log_text: str) -> str:
    if record.get("timed_out"):
        return "TIMEOUT"
    details = record.get("details") or {}
    if isinstance(details, dict) and details.get("status") == "PASS":
        return "PASS"
    if record.get("returncode") == 0:
        # unittest exits 0 for both pass and skip.  Preserve a skip as a
        # distinct status so an NPU-less server cannot look like a full pass.
        if re.search(r"skipped=\d+", log_text):
            return "SKIP"
        return "PASS"
    return "FAIL"


def _run_one(
    case: dict[str, Any],
    run_dir: Path,
    timeout_override: int | None,
) -> dict[str, Any]:
    case_dir = run_dir / case["case"]
    case_dir.mkdir(parents=True, exist_ok=False)
    command = _command_for(case)
    timeout = timeout_override or int(case["timeout"])
    env = os.environ.copy()
    env.update(
        {
            "FLEX_ATTN_CASE_DIR": str(case_dir),
            "TORCH_COMPILE_DEBUG": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(case_dir / "torchinductor_cache"),
            "TRITON_CACHE_DIR": str(case_dir / "triton_cache"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = case_dir / "run.log"
    started = time.monotonic()
    timed_out = False
    returncode: int
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=case_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = 124
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nCASE_TIMEOUT_SECONDS exceeded: {timeout}; subprocess terminated.\n")
    elapsed = round(time.monotonic() - started, 2)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    details = _parse_case_result(log_text)
    output_code_files = _copy_debug_files(case_dir)
    record: dict[str, Any] = {
        "case": case["case"],
        "suite": case["suite"],
        "selector": case.get("selector"),
        "command": command,
        "timeout_seconds": timeout,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "run_log": str(log_path.relative_to(run_dir)),
        "output_code_files": output_code_files,
        "timed_out": timed_out,
        "details": details or {},
        "failure_tail": _failure_tail(log_path),
    }
    record["status"] = _status_for(record, log_text)
    print(
        f"[{record['status']}] {record['case']} "
        f"({record['elapsed_seconds']}s, output_code={len(output_code_files)})",
        flush=True,
    )
    return record


def _selected_cases(suite: str, requested: list[str] | None) -> list[dict[str, Any]]:
    if requested:
        missing = [name for name in requested if name not in CASE_BY_NAME]
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(missing)}")
        return [CASE_BY_NAME[name] for name in requested]
    if suite == "dynamic":
        return list(CUSTOM_CASES)
    if suite == "community":
        return list(COMMUNITY_CASES)
    return list(ALL_CASES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "dynamic", "community"),
        default="all",
        help="test suite to run (default: all)",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="run one named case; repeat the option to select several",
    )
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--run-id", help="run directory name; default: YYYYmmdd_HHMMSS")
    parser.add_argument(
        "--timeout",
        type=int,
        help="override timeout for every selected case (seconds)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop launching new cases after the first FAIL/TIMEOUT",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="only write results.json; do not render REPORT.md automatically",
    )
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    args = parser.parse_args()

    if args.list:
        for case in ALL_CASES:
            selector = f" [{case['selector']}]" if case.get("selector") else ""
            print(f"{case['case']}\t{case['suite']}{selector}")
        return 0

    try:
        selected = _selected_cases(args.suite, args.cases)
    except ValueError as exc:
        parser.error(str(exc))

    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    if run_dir.exists():
        parser.error(f"run directory already exists: {run_dir}; use --run-id with a new name")
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                **_metadata(),
                "run_id": run_id,
                "suite": args.suite,
                "selected_cases": [case["case"] for case in selected],
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "ASCEND_HOME_PATH",
                        "LD_LIBRARY_PATH",
                        "PYTHONPATH",
                        "TORCH_HOME",
                    )
                    if os.environ.get(key)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for case in selected:
        record = _run_one(case, run_dir, args.timeout)
        records.append(record)
        if args.stop_on_failure and record["status"] in {"FAIL", "TIMEOUT"}:
            break
    (run_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.no_report:
        from dump_flex_attention_report import build_report

        markdown, report_data = build_report(run_dir, records)
        (run_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
        (run_dir / "REPORT.json").write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"REPORT={run_dir / 'REPORT.md'}", flush=True)
    print(f"RUN_DIR={run_dir}", flush=True)
    failed = [record for record in records if record["status"] in {"FAIL", "TIMEOUT"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
