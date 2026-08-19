#!/usr/bin/env python3
"""Runtime and source-contract tests for the v2.7.1 dynamic FlexAttention patch.

The runner executes one case per process so every compiled case owns an isolated
torch_compile_debug tree.  Runtime cases deliberately import torch_npu._inductor
before compiling; this is the registration point for the NPU lowering.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


try:
    import torch
    import torch_npu
    import torch_npu._inductor  # noqa: F401  # registers the NPU lowering
    from torch._dynamo.testing import CompileCounterWithBackend
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except Exception as exc:  # pragma: no cover - reported by the runner
    print(f"IMPORT_ERROR={type(exc).__name__}: {exc}", flush=True)
    raise


DEVICE = "npu"
DTYPE = torch.bfloat16
HEAD_DIM = 64
SPARSE_BLOCK = 128
ROOT = Path(__file__).resolve().parent
CASE_DIR = Path(os.environ.get("FLEX_ATTN_CASE_DIR", os.getcwd()))

_pattern: torch.Tensor | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_result_pattern(kind: str, q_len: int, kv_len: int) -> list[list[bool]]:
    """Write a deterministic mask pattern in-place and return a CPU snapshot."""

    global _pattern
    if _pattern is None or tuple(_pattern.shape) != (q_len, kv_len):
        _pattern = torch.empty(
            q_len,
            kv_len,
            device=DEVICE,
            dtype=torch.bool,
        )

    q = torch.arange(q_len, device=DEVICE, dtype=torch.int64)[:, None]
    k = torch.arange(kv_len, device=DEVICE, dtype=torch.int64)[None, :]

    if kind == "all_true":
        value = torch.ones((q_len, kv_len), device=DEVICE, dtype=torch.bool)
    elif kind == "all_false":
        value = torch.zeros((q_len, kv_len), device=DEVICE, dtype=torch.bool)
    elif kind == "one_partial":
        # Exactly one partial block for Q_LEN=KV_LEN=256; all other blocks are
        # full.  This is the smallest non-zero dynamic capacity case.
        value = torch.where((q < 128) & (k < 128), q >= k, q >= 0)
    elif kind == "causal":
        value = q >= k
    elif kind == "sliding":
        value = (q >= k) & ((q - k) < 64)
    elif kind == "mixed":
        # Every row has its diagonal, while block contents are heterogeneous.
        value = (q >= k) | ((q % 3 == 0) & (k % 5 == 0))
    elif kind == "full_partial":
        # Every block has both true and false entries, hence T == C.
        value = (q + k) % 2 == 0
    else:
        raise ValueError(f"unknown mask pattern: {kind}")

    _pattern.copy_(value)
    torch.npu.synchronize()
    return _pattern.detach().cpu().tolist()


def _generic_mask(_b, _h, q_idx, kv_idx):
    if _pattern is None:
        raise RuntimeError("mask pattern has not been initialized")
    return _pattern[q_idx, kv_idx]


def _causal_mask(_b, _h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _make_block_mask(
    *,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    mask_mod: Callable,
) -> BlockMask:
    return create_block_mask(
        mask_mod,
        B=batch,
        H=heads,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=DEVICE,
        BLOCK_SIZE=SPARSE_BLOCK,
    )


def _new_qkv(
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    *,
    requires_grad: bool = False,
):
    q = torch.randn(
        batch,
        heads,
        q_len,
        HEAD_DIM,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        batch,
        heads,
        kv_len,
        HEAD_DIM,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=requires_grad,
    )
    v = torch.randn_like(k, requires_grad=requires_grad)
    return q, k, v


def _dense_reference(q, k, v, mask_cpu):
    mask = torch.tensor(mask_cpu, device=q.device, dtype=torch.bool)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float()).to(q.dtype)


def _assert_close(actual, expected, *, backward: bool = False):
    torch.testing.assert_close(
        actual,
        expected,
        atol=8e-2 if backward else 2e-2,
        rtol=8e-2 if backward else 2e-2,
    )


def _stats(block_mask: BlockMask) -> dict[str, Any]:
    counts = block_mask.kv_num_blocks.detach().to("cpu", dtype=torch.int64)
    total = int(counts.sum().item())
    rows = int(counts.numel())
    max_blocks = int(block_mask.kv_indices.shape[-1])
    capacity = rows * max_blocks
    return {
        "counts": counts.tolist(),
        "rows": rows,
        "max_blocks_per_row": max_blocks,
        "T": total,
        "C": capacity,
        "full_T": (
            int(block_mask.full_kv_num_blocks.detach().to("cpu").sum().item())
            if block_mask.full_kv_num_blocks is not None
            else None
        ),
    }


def _debug_code_files() -> list[Path]:
    return sorted(CASE_DIR.glob("torch_compile_debug/**/output_code.py"))


def _assert_codegen_contract(*, require_backward: bool = False) -> dict[str, Any]:
    files = _debug_code_files()
    assert files, f"no torch_compile_debug/output_code.py under {CASE_DIR}"
    code = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in files)

    offset_call = re.search(r"triton_flex_attention_compact_offsets\.run", code)
    item = re.search(r"\bu\d+\s*=\s*[^\n]*\.item\(\)", code)
    mapping_call = re.search(r"triton_flex_attention_compact_mapping\.run", code)
    mask_alloc = re.search(r"empty_strided\(\(u\d+,\s*128,\s*128\)", code)
    assert offset_call, "offsets kernel call is missing from output code"
    assert item, "DynamicScalar item extraction is missing from output code"
    assert mapping_call, "mapping kernel call is missing from output code"
    assert mask_alloc, "SPARSE_MASK allocation is not based on uT"
    assert offset_call.start() < item.start() < mapping_call.start() < mask_alloc.start(), (
        "output-code order must be offsets -> item -> mapping -> mask allocation"
    )
    assert "TOTAL_FLAT_ENTRIES" not in code
    assert "tl.load(TOTAL_BLOCKS)" not in code
    if require_backward:
        assert "triton_flex_attention_bwd_mask_compact" in code
        assert "dtype=torch.bool" in code

    return {
        "debug_files": [str(path.relative_to(CASE_DIR)) for path in files],
        "offset_line": code[: offset_call.start()].count("\n") + 1,
        "item_line": code[: item.start()].count("\n") + 1,
        "mapping_line": code[: mapping_call.start()].count("\n") + 1,
        "mask_alloc_line": code[: mask_alloc.start()].count("\n") + 1,
        "has_dynamic_mask_allocation": True,
    }


def _expected_mapping(counts: list[list[list[int]]]) -> tuple[list[int], list[int]]:
    rows = []
    blocks = []
    for z in counts:
        for h in z:
            for row, count in enumerate(h):
                rows.extend([row] * int(count))
                blocks.extend(range(int(count)))
    return rows, blocks


def case_source_contracts() -> dict[str, Any]:
    import torch_npu._inductor.kernel.flex_attention as lowering
    import torch_npu._inductor.kernel.flex_attention_metadata as metadata_mod
    import torch_npu._inductor.kernel.flexattention_template as template_mod

    lowering_text = Path(inspect.getfile(lowering)).read_text(encoding="utf-8")
    template_text = Path(inspect.getfile(template_mod)).read_text(encoding="utf-8")
    metadata_text = Path(inspect.getfile(metadata_mod)).read_text(encoding="utf-8")

    required_lowering = (
        "_build_runtime_compact_sparse_mask_offsets",
        "_build_runtime_compact_sparse_mask_mapping",
        "_bind_runtime_total_blocks_as_unbacked_size",
        "DynamicScalar(symbol, (), runtime_total_blocks)",
        "AssertScalar(",
        "pending_fresh_unbacked_symbols",
        "actual_blocks",
    )
    for marker in required_lowering:
        assert marker in lowering_text, f"missing lowering marker: {marker}"

    required_template = (
        "compute_compact_sparse_mask_offsets_kernel",
        "compute_compact_sparse_mask_mapping_kernel",
        'tl.atomic_add(TOTAL_BLOCKS',
        '{{size("FLAT_TO_ROW", 0)}}',
        '{{def_kernel("Q_OFFSETS", "TOTAL_BLOCKS", "KV_NUM_BLKS")}}',
        '{{def_kernel("FLAT_TO_ROW", "FLAT_TO_BLK", "Q_OFFSETS", "KV_NUM_BLKS")}}',
    )
    for marker in required_template:
        assert marker in template_text, f"missing template marker: {marker}"

    for stale in (
        "COMPACT_SPARSE_MASK_TOTAL_BLOCKS",
        "_precompute_compact_sparse_mask_metadata",
        "_wrap_mask_mod_with_compact_sparse_mask_metadata",
        '"TOTAL_FLAT_ENTRIES"',
    ):
        assert stale not in lowering_text, f"stale lowering marker: {stale}"
    assert "_SPARSE_MASK_COMPACT_OPTION_KEYS" not in metadata_text
    assert template_text.count("tl.load(TOTAL_BLOCKS)") == 0

    max_runtime_blocks = torch.iinfo(torch.int32).max // (128 * 128)
    assert max_runtime_blocks == 131071
    assert "guard_leq" in lowering_text
    assert "range_assert = AssertScalar(" in lowering_text

    return {
        "lowering": str(Path(inspect.getfile(lowering))),
        "template": str(Path(inspect.getfile(template_mod))),
        "metadata": str(Path(inspect.getfile(metadata_mod))),
        "max_runtime_blocks_for_128x128": max_runtime_blocks,
        "checks": len(required_lowering) + len(required_template) + 8,
    }


def case_capacity_reuse() -> dict[str, Any]:
    torch.manual_seed(20260818)
    q, k, v = _new_qkv(1, 1, 256, 256)
    counter = CompileCounterWithBackend("inductor")

    def fn(query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
    observations = []
    for kind in ("all_true", "one_partial", "full_partial"):
        print(f"ITERATION_START case=C01 kind={kind}", flush=True)
        mask_cpu = _write_result_pattern(kind, 256, 256)
        block_mask = _make_block_mask(
            batch=1,
            heads=1,
            q_len=256,
            kv_len=256,
            mask_mod=_generic_mask,
        )
        actual = compiled(q, k, v, block_mask)
        print(f"ITERATION_COMPILED case=C01 kind={kind}", flush=True)
        torch.npu.synchronize()
        expected = _dense_reference(q, k, v, mask_cpu)
        _assert_close(actual, expected)
        stats = _stats(block_mask)
        assert stats["T"] in (0, 1, stats["C"]), stats
        observations.append({"kind": kind, **stats})
        print(f"ITERATION_DONE case=C01 kind={kind} T={stats['T']}", flush=True)

    assert [item["T"] for item in observations] == [0, 1, 4]
    assert len({(item["rows"], item["max_blocks_per_row"]) for item in observations}) == 1
    assert counter.frame_count == 1, counter.frame_count
    code = _assert_codegen_contract()
    return {"observations": observations, "frame_count": counter.frame_count, **code}


def case_dynamic_shape_reuse() -> dict[str, Any]:
    torch.manual_seed(20260819)
    counter = CompileCounterWithBackend("inductor")

    def fn(query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
    observations = []
    for batch, heads, q_len, kv_len in (
        # Avoid starting with batch=1: PyTorch's default ShapeEnv policy
        # specializes zero/one dimensions even when dynamic=True, which
        # would make the second batch look like a genuine recompile failure.
        (2, 2, 129, 193),
        (4, 2, 257, 385),
    ):
        print(
            f"ITERATION_START case=C02 shape={(batch, heads, q_len, kv_len)}",
            flush=True,
        )
        q, k, v = _new_qkv(batch, heads, q_len, kv_len)
        block_mask = _make_block_mask(
            batch=batch,
            heads=heads,
            q_len=q_len,
            kv_len=kv_len,
            mask_mod=_causal_mask,
        )
        actual = compiled(q, k, v, block_mask)
        print(
            f"ITERATION_COMPILED case=C02 shape={(batch, heads, q_len, kv_len)}",
            flush=True,
        )
        torch.npu.synchronize()
        q_idx = torch.arange(q_len, device=DEVICE)[:, None]
        kv_idx = torch.arange(kv_len, device=DEVICE)[None, :]
        mask_cpu = (q_idx >= kv_idx).to("cpu").tolist()
        expected = _dense_reference(q, k, v, mask_cpu)
        _assert_close(actual, expected)
        observations.append(
            {
                "shape": [batch, heads, q_len, kv_len, HEAD_DIM],
                **_stats(block_mask),
            }
        )
        print(
            f"ITERATION_DONE case=C02 shape={(batch, heads, q_len, kv_len)}",
            flush=True,
        )

    assert counter.frame_count == 1, counter.frame_count
    code = _assert_codegen_contract()
    return {"observations": observations, "frame_count": counter.frame_count, **code}


def case_non_aligned_modes() -> dict[str, Any]:
    torch.manual_seed(20260820)
    q_len, kv_len = 193, 257
    q, k, v = _new_qkv(1, 1, q_len, kv_len)
    counter = CompileCounterWithBackend("inductor")

    def fn(query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
    observations = []
    for kind in ("causal", "sliding", "mixed", "all_true"):
        print(f"ITERATION_START case=C03 kind={kind}", flush=True)
        mask_cpu = _write_result_pattern(kind, q_len, kv_len)
        block_mask = _make_block_mask(
            batch=1,
            heads=1,
            q_len=q_len,
            kv_len=kv_len,
            mask_mod=_generic_mask,
        )
        actual = compiled(q, k, v, block_mask)
        print(f"ITERATION_COMPILED case=C03 kind={kind}", flush=True)
        torch.npu.synchronize()
        expected = _dense_reference(q, k, v, mask_cpu)
        _assert_close(actual, expected)
        stats = _stats(block_mask)
        assert stats["C"] == 6  # 2 query blocks x 3 KV blocks
        observations.append({"kind": kind, **stats})
        print(f"ITERATION_DONE case=C03 kind={kind} T={stats['T']}", flush=True)

    assert counter.frame_count == 1, counter.frame_count
    code = _assert_codegen_contract()
    return {"observations": observations, "frame_count": counter.frame_count, **code}


def case_non_contiguous_metadata() -> dict[str, Any]:
    torch.manual_seed(20260821)
    q, k, v = _new_qkv(1, 1, 256, 256)
    mask_cpu = _write_result_pattern("one_partial", 256, 256)
    base = _make_block_mask(
        batch=1,
        heads=1,
        q_len=256,
        kv_len=256,
        mask_mod=_generic_mask,
    )

    def strided_last_dim(value):
        padded = torch.zeros(
            *value.shape[:-1],
            value.shape[-1] * 2,
            device=value.device,
            dtype=value.dtype,
        )
        padded[..., ::2].copy_(value)
        return padded[..., ::2]

    kv_num_blocks = strided_last_dim(base.kv_num_blocks)
    kv_indices = strided_last_dim(base.kv_indices)
    assert not kv_num_blocks.is_contiguous()
    assert not kv_indices.is_contiguous()
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks=base.full_kv_num_blocks,
        full_kv_indices=base.full_kv_indices,
        BLOCK_SIZE=SPARSE_BLOCK,
        mask_mod=_generic_mask,
        seq_lengths=(256, 256),
    )

    counter = CompileCounterWithBackend("inductor")

    def fn(query, key, value, mask):
        return flex_attention(query, key, value, block_mask=mask)

    compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
    actual = compiled(q, k, v, block_mask)
    torch.npu.synchronize()
    expected = _dense_reference(q, k, v, mask_cpu)
    _assert_close(actual, expected)

    stats = _stats(block_mask)
    expected_rows, expected_blocks = _expected_mapping(stats["counts"])
    assert len(expected_rows) == stats["T"]
    assert len(expected_blocks) == stats["T"]
    assert expected_rows[:1] == [0]
    assert expected_blocks[:1] == [0]
    assert counter.frame_count == 1
    code = _assert_codegen_contract()
    return {
        "non_contiguous": True,
        "kv_num_blocks_strides": list(kv_num_blocks.stride()),
        "kv_indices_strides": list(kv_indices.stride()),
        "mapping_reference": {
            "flat_to_row": expected_rows,
            "flat_to_blk": expected_blocks,
        },
        "frame_count": counter.frame_count,
        **code,
    }


def case_backward_capacity_and_grad() -> dict[str, Any]:
    torch.manual_seed(20260822)
    counter = CompileCounterWithBackend("inductor")

    def fn(query, key, value, block_mask):
        output = flex_attention(query, key, value, block_mask=block_mask)
        return output.float().square().mean()

    compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
    observations = []
    for kind in ("all_true", "one_partial", "full_partial"):
        print(f"ITERATION_START case=C05 kind={kind}", flush=True)
        q, k, v = _new_qkv(1, 1, 256, 256, requires_grad=True)
        mask_cpu = _write_result_pattern(kind, 256, 256)
        block_mask = _make_block_mask(
            batch=1,
            heads=1,
            q_len=256,
            kv_len=256,
            mask_mod=_generic_mask,
        )

        ref_q = q.detach().clone().requires_grad_(True)
        ref_k = k.detach().clone().requires_grad_(True)
        ref_v = v.detach().clone().requires_grad_(True)
        ref_output = _dense_reference(ref_q, ref_k, ref_v, mask_cpu)
        ref_loss = ref_output.float().square().mean()
        ref_grads = torch.autograd.grad(ref_loss, (ref_q, ref_k, ref_v))

        actual_loss = compiled(q, k, v, block_mask)
        print(f"ITERATION_COMPILED case=C05 kind={kind}", flush=True)
        actual_grads = torch.autograd.grad(actual_loss, (q, k, v))
        torch.npu.synchronize()
        _assert_close(actual_loss, ref_loss, backward=True)
        for actual_grad, expected_grad in zip(actual_grads, ref_grads):
            _assert_close(actual_grad, expected_grad, backward=True)
        observations.append({"kind": kind, **_stats(block_mask)})
        print(
            f"ITERATION_DONE case=C05 kind={kind} T={observations[-1]['T']}",
            flush=True,
        )

    assert [item["T"] for item in observations] == [0, 1, 4]
    assert counter.frame_count == 1, counter.frame_count
    code = _assert_codegen_contract(require_backward=True)
    return {"observations": observations, "frame_count": counter.frame_count, **code}


def case_non_default_stream() -> dict[str, Any]:
    torch.manual_seed(20260823)
    q, k, v = _new_qkv(1, 1, 256, 256)
    counter = CompileCounterWithBackend("inductor")

    def fn(query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        mask_cpu = _write_result_pattern("one_partial", 256, 256)
        block_mask = _make_block_mask(
            batch=1,
            heads=1,
            q_len=256,
            kv_len=256,
            mask_mod=_generic_mask,
        )
        actual = compiled(q, k, v, block_mask)
    stream.synchronize()
    torch.npu.synchronize()
    expected = _dense_reference(q, k, v, mask_cpu)
    _assert_close(actual, expected)
    assert counter.frame_count == 1
    code = _assert_codegen_contract()
    return {
        "stream_mode": "explicit_non_default",
        "stream": str(stream),
        "T": _stats(block_mask)["T"],
        "frame_count": counter.frame_count,
        **code,
    }


def case_codegen_order() -> dict[str, Any]:
    torch.manual_seed(20260824)
    q, k, v = _new_qkv(1, 1, 256, 256)
    mask_cpu = _write_result_pattern("one_partial", 256, 256)
    block_mask = _make_block_mask(
        batch=1,
        heads=1,
        q_len=256,
        kv_len=256,
        mask_mod=_generic_mask,
    )

    compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)
    actual = compiled(q, k, v, block_mask=block_mask)
    torch.npu.synchronize()
    _assert_close(actual, _dense_reference(q, k, v, mask_cpu))
    return {"stats": _stats(block_mask), **_assert_codegen_contract()}


CASES = {
    "S00_source_contracts": case_source_contracts,
    "C01_capacity_reuse": case_capacity_reuse,
    "C02_dynamic_shape_reuse": case_dynamic_shape_reuse,
    "C03_non_aligned_modes": case_non_aligned_modes,
    "C04_non_contiguous_metadata": case_non_contiguous_metadata,
    "C05_backward_capacity_and_grad": case_backward_capacity_and_grad,
    "C06_non_default_stream": case_non_default_stream,
    "C07_codegen_order": case_codegen_order,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    args = parser.parse_args()

    result: dict[str, Any] = {
        "case": args.case,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "npu_available": bool(torch.npu.is_available()),
    }
    try:
        assert torch.npu.is_available(), "NPU is not available"
        result.update(_jsonable(CASES[args.case]()))
        result["status"] = "PASS"
    except Exception as exc:  # runner records the complete traceback in run.log
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    print("CASE_RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
