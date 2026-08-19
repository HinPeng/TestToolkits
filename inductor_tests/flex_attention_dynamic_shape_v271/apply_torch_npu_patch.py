#!/usr/bin/env python3
"""Apply the v2.7.1 FlexAttention patch to an installed torch_npu wheel.

The patch was produced from a torch_npu source checkout, while a wheel is not
a git repository.  This helper therefore extracts only the ``torch_npu/``
sections and applies them with the system ``patch`` utility.  The test file
section from the original patch is intentionally skipped because the complete
test suite is shipped beside this script.

Examples:

    python apply_torch_npu_patch.py --dry-run
    python apply_torch_npu_patch.py
    python apply_torch_npu_patch.py --package-root /path/to/site-packages
    python apply_torch_npu_patch.py --reverse
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_VERSION = "2.7.1"
PATCHED_MARKERS = {
    "_inductor/kernel/flex_attention.py": (
        "_build_runtime_compact_sparse_mask_offsets",
        "DynamicScalar(symbol, (), runtime_total_blocks)",
    ),
    "_inductor/kernel/flexattention_template.py": (
        "compute_compact_sparse_mask_offsets_kernel",
        "compute_compact_sparse_mask_mapping_kernel",
    ),
    "_inductor/kernel/flex_attention_metadata.py": (
        "infer_eager_block_mask_kernel_options",
    ),
}
PATCHED_ABSENT_MARKERS = {
    "_inductor/kernel/flex_attention.py": (
        "COMPACT_SPARSE_MASK_TOTAL_BLOCKS",
        '"TOTAL_FLAT_ENTRIES"',
    ),
}
PATCH_PATH_RE = re.compile(r"^diff --git a/torch_npu/[^ ]+ b/torch_npu/[^ ]+")


class PatchError(RuntimeError):
    """An actionable patch application error."""


def _package_dir(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.name == "torch_npu" and (candidate / "__init__.py").is_file():
            return candidate
        package = candidate / "torch_npu"
        if (package / "__init__.py").is_file():
            return package
        raise PatchError(
            f"--package-root does not contain torch_npu/__init__.py: {candidate}"
        )

    try:
        spec = importlib.util.find_spec("torch_npu")
    except Exception as exc:  # pragma: no cover - depends on broken environments
        raise PatchError(f"cannot inspect torch_npu import path: {exc}") from exc
    if spec is None or not spec.submodule_search_locations:
        raise PatchError(
            "torch_npu is not importable; activate the Python environment "
            "that contains torch_npu 2.7.1 or pass --package-root"
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _installed_version(package_dir: Path) -> str:
    try:
        return importlib.metadata.version("torch_npu")
    except importlib.metadata.PackageNotFoundError:
        pass

    for candidate in (
        package_dir / "version.txt",
        package_dir.parent / "version.txt",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()

    version_py = package_dir / "version.py"
    if version_py.is_file():
        text = version_py.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?:__version__|VERSION)\s*=\s*[\"']([^\"']+)", text)
        if match:
            return match.group(1)
    return "unknown"


def _base_version(version: str) -> str:
    match = re.search(r"\d+\.\d+\.\d+", version)
    return match.group(0) if match else version


def _extract_torch_npu_sections(patch_file: Path) -> str:
    text = patch_file.read_text(encoding="utf-8")
    sections: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current and PATCH_PATH_RE.match(current[0]):
            sections.append("".join(current))

    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            current = [line]
        elif current:
            current.append(line)
    flush()

    if not sections:
        raise PatchError(f"no torch_npu diff sections found in {patch_file}")
    return "".join(sections)


def _is_patched(package_dir: Path) -> bool:
    for relative, markers in PATCHED_MARKERS.items():
        path = package_dir / relative
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker not in text for marker in markers):
            return False
    for relative, markers in PATCHED_ABSENT_MARKERS.items():
        path = package_dir / relative
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in markers):
            return False
    return True


def _verify_patched(package_dir: Path) -> None:
    if not _is_patched(package_dir):
        raise PatchError(
            "patch command completed but the post-apply source markers are "
            f"incomplete under {package_dir}"
        )


def _run_patch(
    package_parent: Path,
    filtered_patch: Path,
    *,
    reverse: bool,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    command = ["patch", "-p1", "--batch"]
    command.append("--reverse" if reverse else "--forward")
    if dry_run:
        command.append("--dry-run")
    command.extend(["--input", str(filtered_patch)])
    return subprocess.run(
        command,
        cwd=package_parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _print_patch_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout.rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-file",
        type=Path,
        default=Path(__file__).with_name("flex-attention-dynamic-shape-v2.7.1.patch"),
        help="original git-format patch (default: beside this script)",
    )
    parser.add_argument(
        "--package-root",
        help="site-packages directory or its torch_npu directory; default: import path",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="allow a torch_npu base version other than 2.7.1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate patch applicability without changing installed files",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="reverse the patch (only use after verifying the target package)",
    )
    args = parser.parse_args()

    try:
        package_dir = _package_dir(args.package_root)
        version = _installed_version(package_dir)
        if not args.allow_version_mismatch and _base_version(version) != EXPECTED_VERSION:
            raise PatchError(
                f"torch_npu version is {version!r}, expected base version {EXPECTED_VERSION}; "
                "use --allow-version-mismatch only after checking the source baseline"
            )
        patch_file = args.patch_file.expanduser().resolve()
        if not patch_file.is_file():
            raise PatchError(f"patch file does not exist: {patch_file}")
        filtered_text = _extract_torch_npu_sections(patch_file)

        print(f"torch_npu={version} ({package_dir})")
        print(f"patch={patch_file}")
        print("patch_sections=torch_npu only; bundled test section skipped")

        if not args.reverse and _is_patched(package_dir):
            print("status=ALREADY_APPLIED")
            return 0
        if args.reverse and not _is_patched(package_dir):
            raise PatchError("cannot reverse: expected patched source markers are absent")

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".patch", prefix="torch_npu_flex_", delete=False
        ) as temporary:
            temporary.write(filtered_text)
            filtered_patch = Path(temporary.name)

        try:
            check = _run_patch(
                package_dir.parent,
                filtered_patch,
                reverse=args.reverse,
                dry_run=True,
            )
            _print_patch_output(check)
            if check.returncode != 0:
                raise PatchError(
                    "patch preflight failed; installed files were not modified. "
                    "The wheel may not match the v2.7.1 source baseline."
                )
            if args.dry_run:
                print("status=DRY_RUN_OK")
                return 0

            applied = _run_patch(
                package_dir.parent,
                filtered_patch,
                reverse=args.reverse,
                dry_run=False,
            )
            _print_patch_output(applied)
            if applied.returncode != 0:
                raise PatchError("patch command failed; inspect the output above")
            if args.reverse:
                print("status=REVERSED")
            else:
                _verify_patched(package_dir)
                print("status=APPLIED")
            return 0
        finally:
            filtered_patch.unlink(missing_ok=True)
    except (OSError, PatchError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
