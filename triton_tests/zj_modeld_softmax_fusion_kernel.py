import time

import torch
import torch_npu
import triton
import triton.language as tl


SEQ_Q = 424
NUM_HEADS = 16
KV_LEN = 8000
MASK_SCALE = 10000.0
XNUMEL = SEQ_Q * NUM_HEADS
R0_NUMEL = KV_LEN


# Best config reported by autotune:
BEST_CONFIG = {
    "XBLOCK": 16,
    "R0_BLOCK": 128,
    "compile_mode": "simt_only",
    "multibuffer": True,
    "num_warps": 32,
    "num_ctas": 1,
    "num_stages": 2,
}


@triton.jit
def triton_red_fused__softmax__to_copy_mul_rsub_sub_0(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr2,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 16

    tmp_max = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel

        lhs = tl.load(
            in_ptr0 + (r0_index + r0_numel * x3),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        rhs = tl.load(
            in_ptr1 + (r0_index + r0_numel * x3),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        mask = tl.load(
            in_ptr2 + (r0_index + r0_numel * x1),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0,
        )
        score_delta = lhs - rhs
        inverted_mask = tl.full([1, 1], 1, tl.int32) - mask
        mask_bias = inverted_mask.to(tl.float32) * 10000.0
        masked_scores = score_delta - mask_bias
        tmp_max = tl.where(
            r0_mask & xmask,
            tl.maximum(tmp_max, tl.broadcast_to(masked_scores, [XBLOCK, R0_BLOCK])),
            tmp_max,
        )

    row_max = tl.max(tmp_max, axis=1)[:, None]

    tmp_sum = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel

        lhs = tl.load(
            in_ptr0 + (r0_index + r0_numel * x3),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        rhs = tl.load(
            in_ptr1 + (r0_index + r0_numel * x3),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        mask = tl.load(
            in_ptr2 + (r0_index + r0_numel * x1),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0,
        )
        score_delta = lhs - rhs
        inverted_mask = tl.full([1, 1], 1, tl.int32) - mask
        mask_bias = inverted_mask.to(tl.float32) * 10000.0
        masked_scores = score_delta - mask_bias
        exp_scores = tl.exp(masked_scores - row_max)
        tmp_sum = tl.where(
            r0_mask & xmask,
            tmp_sum + tl.broadcast_to(exp_scores, [XBLOCK, R0_BLOCK]),
            tmp_sum,
        )

    row_sum = tl.sum(tmp_sum, 1)[:, None]

    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel

        lhs = tl.load(
            in_ptr0 + (r0_index + r0_numel * x3),
            xmask & r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        rhs = tl.load(
            in_ptr1 + (r0_index + r0_numel * x3),
            xmask & r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        mask = tl.load(
            in_ptr2 + (r0_index + r0_numel * x1),
            xmask & r0_mask,
            eviction_policy="evict_last",
            other=0,
        )
        score_delta = lhs - rhs
        inverted_mask = tl.full([1, 1], 1, tl.int32) - mask
        mask_bias = inverted_mask.to(tl.float32) * 10000.0
        masked_scores = score_delta - mask_bias
        probs = tl.exp(masked_scores - row_max) / row_sum
        out = probs * mask.to(tl.float32)
        tl.store(out_ptr2 + (r0_index + r0_numel * x3), out, xmask & r0_mask)


def reference_eager(lhs, rhs, mask):
    score_delta = lhs - rhs
    inverted_mask = 1 - mask
    mask_bias = inverted_mask.to(torch.float32) * MASK_SCALE
    masked_scores = score_delta - mask_bias
    probs = torch.softmax(masked_scores, dim=-1)
    return probs * mask.to(torch.float32)


def make_inputs(device="npu"):
    lhs = torch.randn((424, 16, 8000), device=device, dtype=torch.float32)
    rhs = torch.randn((424, 16, 8000), device=device, dtype=torch.float32)
    mask = torch.randint(0, 2, (424, 1, 8000), device=device, dtype=torch.int32)
    return lhs, rhs, mask


def launch_kernel(lhs, rhs, mask, out=None):
    out = torch.empty_like(lhs) if out is None else out
    grid = lambda meta: (triton.cdiv(XNUMEL, meta["XBLOCK"]),)

    # `split_axis` / `split_blocks` are autotune metadata rather than plain
    # triton launch kwargs. Their effect is already reflected in the selected
    # launch shape (`grid`, `XBLOCK`, `R0_BLOCK`).
    triton_red_fused__softmax__to_copy_mul_rsub_sub_0[grid](
        lhs,
        rhs,
        mask,
        out,
        XNUMEL,
        R0_NUMEL,
        XBLOCK=BEST_CONFIG["XBLOCK"],
        R0_BLOCK=BEST_CONFIG["R0_BLOCK"],
        num_warps=BEST_CONFIG["num_warps"],
        num_stages=BEST_CONFIG["num_stages"],
        num_ctas=BEST_CONFIG["num_ctas"],
        compile_mode=BEST_CONFIG["compile_mode"],
        multibuffer=BEST_CONFIG["multibuffer"],
    )
    return out


def benchmark(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    elapsed_s = time.perf_counter() - start
    return elapsed_s * 1e3 / repeat


if __name__ == "__main__":
    torch.npu.set_device(0)

    lhs, rhs, mask = make_inputs()
    ref = reference_eager(lhs, rhs, mask)
    out = torch.empty_like(lhs)
    out = launch_kernel(lhs, rhs, mask, out)
    torch.testing.assert_close(ref, out, atol=1e-4, rtol=1e-4)

    triton_ms = benchmark(lambda: launch_kernel(lhs, rhs, mask, out))
    eager_ms = benchmark(lambda: reference_eager(lhs, rhs, mask))

    print("best config:", BEST_CONFIG)
    print(f"eager: {eager_ms:.3f} ms")
    print(f"triton: {triton_ms:.3f} ms")
    print(f"speedup: {eager_ms / triton_ms:.3f}x")
