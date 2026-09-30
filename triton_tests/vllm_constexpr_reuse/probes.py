"""Derived controls, explicitly not frozen production vLLM kernels."""

import triton
import triton.language as tl


@triton.jit
def renamed_pointwise(a, b, result, factor, length, work, CHUNK: tl.constexpr):
    lane = tl.program_id(axis=0)
    lanes = tl.num_programs(axis=0)
    for part in range(lane, work, lanes):
        start = part * CHUNK
        idx = start + tl.arange(0, CHUNK)
        live = idx < length
        lhs = tl.load(a + idx, mask=live)
        rhs = tl.load(b + idx, mask=live)
        value = lhs * factor + rhs
        tl.store(result + idx, value, mask=live)


@triton.jit(do_not_specialize=["rows", "width", "limit"])
def external_bound_copy(a, result, rows, width, limit, CHUNK: tl.constexpr):
    lane = tl.program_id(0)
    lanes = tl.num_programs(0)
    pieces = tl.cdiv(width, CHUNK)
    for item in tl.range(lane, limit, lanes):
        row = item // pieces
        piece = item - row * pieces
        col = piece * CHUNK + tl.arange(0, CHUNK)
        value = tl.load(a + row * width + col, mask=col < width, other=0)
        tl.store(result + row * width + col, value, mask=col < width)
