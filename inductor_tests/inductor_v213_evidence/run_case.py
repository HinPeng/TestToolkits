"""Run one evidence case and write a machine-readable summary.

The shell entrypoint sets TORCH_LOGS/TORCH_COMPILE_DEBUG before importing
PyTorch.  This file focuses on the semantic case, correctness, metadata, and
optional Wrapper FX IR capture.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_map

from cases.vector_cases import CASES, make_inputs


RELEVANT_ENV_KEYS = (
    "TORCH_LOGS",
    "TORCH_LOGS_OUT",
    "TORCH_COMPILE_DEBUG",
    "TORCH_COMPILE_DEBUG_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_ENABLED_METRIC_TABLES",
    "TRITON_CACHE_DIR",
    "CUDA_VISIBLE_DEVICES",
)


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "is_contiguous": value.is_contiguous(),
    }


def _tree_tensor_metadata(value: Any) -> list[dict[str, Any]]:
    return [
        _tensor_metadata(item)
        for item in tree_flatten(value)[0]
        if isinstance(item, torch.Tensor)
    ]


def _clone_tree(value: Any) -> Any:
    return tree_map(lambda item: item.clone() if isinstance(item, torch.Tensor) else item, value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)  # type: ignore[attr-defined]


def _device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"requested": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device.index is None else device.index
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "index": index,
                "name": props.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": props.total_memory,
                "device_count": torch.cuda.device_count(),
            }
        )
    elif device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        info["device_count"] = torch.npu.device_count()
    return info


def _torch_info() -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception as exc:  # pragma: no cover - depends on remote runtime
        triton_version = f"unavailable: {exc}"
    return {
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": torch.version.hip,
        "triton_version": triton_version,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def _config_options(variant: str) -> dict[str, Any]:
    common = {
        # Stable ordering and an isolated graph artifact for each run.
        "compile_threads": 1,
        "fx_graph_cache": False,
        "graph_partition": False,
        "cpp_wrapper": False,
    }
    if variant == "baseline":
        return common
    if variant == "fxir":
        return {**common, "fx_wrapper": True}
    if variant == "reindex_on":
        return {**common, "loop_reindexing_after_fusion": True}
    if variant == "reindex_off":
        return {**common, "loop_reindexing_after_fusion": False}
    if variant == "partition":
        return {**common, "graph_partition": True}
    if variant == "autotune":
        return {
            **common,
            "max_autotune": True,
            "max_autotune_pointwise": True,
        }
    raise ValueError(f"Unknown variant: {variant}")


class WrapperFxCapture:
    """Persist WrapperFxCodegen's in-memory GraphModule for a run."""

    def __init__(self, output_dir: Path, enabled: bool):
        self.output_dir = output_dir
        self.enabled = enabled
        self._converter_cls: Any = None
        self._original: Any = None
        self._index = 0

    def __enter__(self) -> "WrapperFxCapture":
        if not self.enabled:
            return self

        from torch._inductor.codegen.wrapper_fxir import FxConverter

        self._converter_cls = FxConverter
        self._original = FxConverter.generate

        def generate(converter: Any) -> torch.fx.GraphModule:
            gm = self._original(converter)
            index = self._index
            self._index += 1
            (self.output_dir / f"wrapper_fx_{index}.py").write_text(
                gm.code, encoding="utf-8"
            )
            (self.output_dir / f"wrapper_fx_{index}_graph.txt").write_text(
                str(gm.graph), encoding="utf-8"
            )
            # Branch GraphModules are attached through get_attr.  Save their
            # code separately so a parent/child graph can be read offline.
            for name, module in gm.named_modules():
                if not name or not isinstance(module, torch.fx.GraphModule):
                    continue
                safe_name = name.replace(".", "_")
                (self.output_dir / f"wrapper_fx_{index}_{safe_name}.py").write_text(
                    module.code, encoding="utf-8"
                )
            return gm

        FxConverter.generate = generate
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any) -> None:
        if self._converter_cls is not None and self._original is not None:
            self._converter_cls.generate = self._original


def _metrics_snapshot() -> dict[str, Any]:
    from torch._inductor import metrics
    from torch._dynamo.utils import counters

    counter_snapshot: dict[str, dict[str, Any]] = {}
    for group, values in counters.items():
        counter_snapshot[str(group)] = {str(key): value for key, value in values.items()}
    return {
        "generated_kernel_count": metrics.generated_kernel_count,
        "num_bytes_accessed": metrics.num_bytes_accessed,
        "counters": counter_snapshot,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this case on a CUDA host")
    if device.type == "npu" and (
        not hasattr(torch, "npu") or not torch.npu.is_available()  # type: ignore[attr-defined]
    ):
        raise RuntimeError("NPU is unavailable; this first batch targets CUDA")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    spec, inputs = make_inputs(args.case, device, args.shape_mode, dtype)
    options = _config_options(args.variant)
    input_metadata = [_tensor_metadata(item) for item in inputs]

    from torch._inductor import metrics

    try:
        from torch import _dynamo

        _dynamo.reset()
    except Exception:
        pass
    metrics.reset()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "case_id": spec.case_id,
        "case_title": spec.title,
        "source_reference": spec.source_reference,
        "variant": args.variant,
        "shape_mode": args.shape_mode,
        "dtype": args.dtype,
        "device": _device_info(device),
        "torch": _torch_info(),
        "config_options": options,
        "seed": args.seed,
        "repeat": args.repeat,
        "input_metadata": input_metadata,
        "relevant_environment": {
            key: os.environ.get(key) for key in RELEVANT_ENV_KEYS if os.environ.get(key)
        },
        "argv": sys.argv,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "input_metadata.json", input_metadata)

    eager_inputs = _clone_tree(inputs)
    compiled_inputs = _clone_tree(inputs)
    with torch.no_grad():
        eager_output = spec.fn(*eager_inputs)

    capture = WrapperFxCapture(output_dir, args.variant == "fxir")
    timings_ms: list[float] = []
    try:
        with torch.no_grad(), capture:
            compiled = torch.compile(spec.fn, options=options)
            start = time.perf_counter()
            compiled_output = compiled(*compiled_inputs)
            _sync(device)
            first_call_ms = (time.perf_counter() - start) * 1000

            torch.testing.assert_close(eager_output, compiled_output)
            for _ in range(args.repeat):
                repeat_inputs = _clone_tree(inputs)
                start = time.perf_counter()
                output = compiled(*repeat_inputs)
                _sync(device)
                timings_ms.append((time.perf_counter() - start) * 1000)
                torch.testing.assert_close(eager_output, output)

        manifest.update(
            {
                "status": "passed",
                "first_call_ms": first_call_ms,
                "repeat_call_ms": timings_ms,
                "eager_output_metadata": _tree_tensor_metadata(eager_output),
                "compiled_output_metadata": _tree_tensor_metadata(compiled_output),
                "metrics": _metrics_snapshot(),
            }
        )
        _write_json(output_dir / "run_summary.json", manifest)
        _write_json(output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, default=_json_default))
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "metrics": _metrics_snapshot(),
            }
        )
        _write_json(output_dir / "run_summary.json", manifest)
        _write_json(output_dir / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument(
        "--variant",
        choices=("baseline", "fxir", "reindex_on", "reindex_off", "partition", "autotune"),
        default="baseline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shape-mode", choices=("trace", "canonical"), default="trace")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
