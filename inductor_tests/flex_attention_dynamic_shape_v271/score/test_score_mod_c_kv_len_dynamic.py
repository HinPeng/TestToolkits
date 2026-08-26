"""Category C: dynamic KV_LEN (KV varies, B=2, Q=512 fixed) × 11 score_mods.

Mirrors dynamic-fix/test_flex_attention_c_batch_envelope.py but adds 11
score_mods. Q != KV (cross-attention-like shape) to test KV axis independently.

Pattern:
- Each KV_LEN gets an exact, broadcastable ``B=1, H=1`` BlockMask captured
  by ``functools.partial`` together with ``score_mod``.
- All compiled partials share one ``CompileCounterWithBackend`` backend;
  Dynamo cache reuse => ``frame_count == 1``.
- Dense SDPA reference (fp32 math); fp16 uses project-wide 5e-3, bf16 uses
  1e-2 across all KV (bf16 7-bit mantissa accumulates ~1e-2 element-wise
  error vs fp32 reference even at KV=512/1024).
- Forward only.

Run:
    pytest test_score_mod_c_kv_len_dynamic.py -v
"""
import pytest
import torch

from torch.nn.attention.flex_attention import create_block_mask, noop_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_score_mod_dynamic_common import (
    npu_device,            # noqa: F401
    DEFAULT_H_Q,
    DEFAULT_D,
    ALL_SCORE_MOD_NAMES,
    resolve_score_mod,
    dense_reference,
    compile_flex_with,
    assert_close,
    assert_frame_count,
    _get_head_offset,
)


# KV_LEN spans 256..2048 (aligned multiples of 128).
KV_LEN_SHAPES = [256, 512, 1024, 2048]
# Fixed B and Q (exclude 1 to avoid Dynamo 0/1 specialization).
B_FIXED = 2
Q_FIXED = 512
# bf16 has only 7 mantissa bits (~4e-3 relative precision). The dense fp32
# reference vs NPU bf16 kernel accumulates ~1e-2 element-wise error, exceeding
# the project-wide 5e-3 tolerance that was sized for long-reduction LLM shapes
# (base/test_flex_attention.py runs reductions of 26k+). fp16 passes 5e-3, so
# only bf16 needs the looser tolerance; apply uniformly across all KV.
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2


@pytest.mark.parametrize("score_mod_name", ALL_SCORE_MOD_NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c_kv_len_dynamic(npu_device, dtype, score_mod_name):
    """C: KV_LEN in [256,512,1024,2048] dynamic, B=2, Q=512, score_mod × dtype.

    Validates one reused graph across KV_LEN variations (Q != KV) for each
    score_mod. Cross-attention-like shape.
    """
    H, D = DEFAULT_H_Q, DEFAULT_D
    counter = CompileCounterWithBackend("inductor")

    head_offset_buffer = None
    if score_mod_name == "head_offset":
        head_offset_buffer = _get_head_offset(H, npu_device, dtype)
    score_mod = resolve_score_mod(
        score_mod_name, num_heads=H, device=npu_device, dtype=dtype
    )

    # bf16 uses looser tolerance (see BF16_ATOL doc); fp16 uses default 5e-3.
    if dtype == torch.bfloat16:
        atol, rtol = BF16_ATOL, BF16_RTOL
    else:
        atol, rtol = None, None  # use assert_close defaults (SD_ATOL/SD_RTOL)

    for KV in KV_LEN_SHAPES:
        bm = create_block_mask(
            noop_mask, B=1, H=1, Q_LEN=Q_FIXED, KV_LEN=KV, device=npu_device,
        )
        compiled = compile_flex_with(counter, score_mod=score_mod, block_mask=bm)
        q = torch.randn(B_FIXED, H, Q_FIXED, D, device=npu_device, dtype=dtype)
        k = torch.randn(B_FIXED, H, KV, D, device=npu_device, dtype=dtype)
        v = torch.randn(B_FIXED, H, KV, D, device=npu_device, dtype=dtype)
        actual = compiled(q, k, v)
        expected = dense_reference(
            q, k, v, score_mod_name=score_mod_name,
            head_offset_buffer=head_offset_buffer,
        )
        if atol is not None:
            assert_close(
                actual, expected, f"C {score_mod_name} KV={KV} {dtype}",
                atol=atol, rtol=rtol,
            )
        else:
            assert_close(actual, expected, f"C {score_mod_name} KV={KV} {dtype}")

    assert_frame_count(counter, f"C {score_mod_name}")
