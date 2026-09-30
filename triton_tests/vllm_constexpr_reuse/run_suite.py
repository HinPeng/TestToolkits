"""Run off/on in fresh processes and compare every output bit for bit."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from snapshot import verify
from catalog import NPU_TEST_FILES, validate_operator_coverage

ROOT = Path(__file__).resolve().parent


def validate_report(report, *, require_positive=True):
    if report["exitstatus"]:
        raise ValueError(f"pytest failed: {report['exitstatus']}")
    cases = report["cases"]
    if not cases or any(not value["launches"] or not value["outputs"] for value in cases.values()):
        raise ValueError("no complete NPU operator evidence (skips do not count as success)")
    if report["outcomes"].keys() != cases.keys():
        raise ValueError("some selected operator tests have no launch evidence")
    for name in cases:
        outcomes = report["outcomes"].get(name, {})
        if any(outcomes.get(phase) != "passed" for phase in ("setup", "call", "teardown")):
            raise ValueError(f"incomplete operator test: {name}: {outcomes}")
    if require_positive and report["strict"] and not any(value["reuse_assertions"] for value in cases.values()):
        raise ValueError("strict selection contains no positive D/T reuse assertion")


def compare_reports(off, on):
    validate_report(off)
    validate_report(on)
    if off["mode"] != "0" or on["mode"] != "1":
        raise ValueError("expected independently launched off/on reports")
    if off.get("versions") != on.get("versions") or off.get("source_revision") != on.get("source_revision"):
        raise ValueError("off/on toolchain, device or source revision differs")
    if off["cases"].keys() != on["cases"].keys():
        raise ValueError("off/on case sets differ")
    differences = [name for name in off["cases"]
                   if off["cases"][name]["outputs"] != on["cases"][name]["outputs"]]
    if differences:
        raise ValueError("off/on output bytes differ: " + ", ".join(differences))
    return len(off["cases"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "off", "on"), default="both")
    parser.add_argument("--strict-reuse", action="store_true", help="Assert reuse in the on process")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--select", default="", help="pytest -k expression")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args(argv)
    if args.strict_reuse and args.mode == "off":
        parser.error("--strict-reuse needs --mode on or both")
    verify()
    # Never consume an earlier report or warmed cache as this run's evidence.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    collection = output / "collection.json"
    collect_command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *NPU_TEST_FILES,
                       f"--reuse-collect={collection}"]
    if args.select:
        collect_command.extend(["-k", args.select])
    collected = subprocess.run(collect_command, cwd=ROOT, check=False)
    if collected.returncode:
        return collected.returncode
    node_ids = json.loads(collection.read_text())
    if not node_ids:
        raise ValueError("no selected scenarios")
    modes = ("off", "on") if args.mode == "both" else (args.mode,)
    reports = {}
    for mode in modes:
        env = os.environ.copy()
        env["TRITON_ASCEND_ENABLE_DYNAMIC_REUSE"] = "0" if mode == "off" else "1"
        env["PYTHONHASHSEED"] = "0"
        env.pop("TRITON_INTERPRET", None)
        combined = None
        for index, node_id in enumerate(node_ids):
            directory = output / mode / f"{index:03d}"
            directory.mkdir(parents=True)
            env["TRITON_CACHE_DIR"] = str(directory / "cache")
            report = directory / "report.json"
            command = [sys.executable, "-m", "pytest", "-q", "-ra", node_id,
                       "--require-npu", f"--npu-device={args.device}", f"--reuse-report={report}"]
            if args.strict_reuse and mode == "on":
                command.append("--verify-reuse")
            print(f"Running {mode} {index + 1}/{len(node_ids)}: {node_id}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                return result.returncode
            evidence = json.loads(report.read_text())
            validate_report(evidence, require_positive=False)
            if combined is None:
                combined = evidence
            else:
                combined["cases"].update(evidence["cases"])
                combined["outcomes"].update(evidence["outcomes"])
        reports[mode] = combined
        if not args.select:
            combined["operators_observed"] = validate_operator_coverage(combined)
        (output / f"{mode}.json").write_text(json.dumps(combined, indent=2) + "\n")
        validate_report(reports[mode])
    if len(reports) == 2:
        count = compare_reports(reports["off"], reports["on"])
        print(f"PASS: {count} operator scenarios, off/on outputs identical; reports: {output}")
    else:
        print(f"PASS: numerical checks completed; report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
