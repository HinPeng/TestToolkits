"""Category D: combined dynamic B x Q x KV × 11 score_mods (forward + backward).

Mirrors dynamic-fix/test_flex_attention_d_capacity_envelope.py but adds 11
score_mods and adds backward grad checks (q/k/v grads vs dense reference grads).

Pattern:
- Each runtime (B, Q, KV) gets an exact, broadcastable ``B=1, H=1`` BlockMask
  captured by ``functools.partial`` together with ``score_mod``.
- All compiled partials share one ``CompileCounterWithBackend`` backend;
  Dynamo cache reuse => ``frame_count == 1``.
- Dense SDPA reference (fp32) for both forward and backward grads.
- atol=5e-3, rtol=5e-3 (forward); same tolerance for grads (per user spec).

B excludes 1 (Dynamo 0/1 specialization). Q/KV aligned multiples of 128
(avoid capacity bucket edges).

Run:
    pytest test_score_mod_d_b_q_kv_joint_dynamic.py -v
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
    assert_close_with_details,
    assert_frame_count,
    assert_grad_finite,
    _get_head_offset,
)


# (B, Q, KV) combos with all three axes varying together.
# B excludes 1 (0/1 specialization); Q/KV aligned multiples of 128.
# First shape uses Q != KV to avoid Q=KV shape specialization.
BQKV_SHAPES = [
    (2, 256, 384),
    (4, 512, 512),
    (8, 1024, 1024),
    (2, 512, 1024),   # Q != KV cross-attn shape
    (4, 1024, 512),
]


@pytest.mark.parametrize("score_mod_name", ALL_SCORE_MOD_NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_d_b_q_kv_joint_dynamic(npu_device, dtype, score_mod_name):
    """D: (B,Q,KV) all dynamic, score_mod × dtype. Forward + backward.

    Validates one reused graph across B+Q+KV joint variations for each
    score_mod, plus q/k/v backward grads match dense reference.
    """
    H, D = DEFAULT_H_Q, DEFAULT_D
    counter = CompileCounterWithBackend("inductor")

    head_offset_buffer = None
    if score_mod_name == "head_offset":
        head_offset_buffer = _get_head_offset(H, npu_device, dtype)
    score_mod = resolve_score_mod(
        score_mod_name, num_heads=H, device=npu_device, dtype=dtype
    )

    for B, Q, KV in BQKV_SHAPES:
        bm = create_block_mask(
            noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device,
        )
        compiled = compile_flex_with(counter, score_mod=score_mod, block_mask=bm)

        # requires_grad for backward
        q = torch.randn(B, H, Q, D, device=npu_device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, KV, D, device=npu_device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, KV, D, device=npu_device, dtype=dtype, requires_grad=True)

        # Forward
        actual = compiled(q, k, v)
        torch.npu.synchronize()
        with torch.no_grad():
            expected = dense_reference(
                q, k, v, score_mod_name=score_mod_name,
                head_offset_buffer=head_offset_buffer,
            )
        torch.npu.synchronize()
        assert_close(actual, expected, f"D {score_mod_name} B={B} Q={Q} KV={KV} fwd {dtype}")

        # Backward: compare compiled grads to dense reference grads
        # Use same grad_out for both compiled and reference
        grad_out = torch.randn_like(actual)
        actual.backward(grad_out)
        torch.npu.synchronize()
        compiled_q_grad = q.grad.clone()
        compiled_k_grad = k.grad.clone()
        compiled_v_grad = v.grad.clone()

        # Sanity: grads finite and shape-correct (check before clearing q/k/v.grad)
        assert_grad_finite(q, "q", f"D {score_mod_name} B={B} Q={Q} KV={KV}")
        assert_grad_finite(k, "k", f"D {score_mod_name} B={B} Q={Q} KV={KV}")
        assert_grad_finite(v, "v", f"D {score_mod_name} B={B} Q={Q} KV={KV}")

        # Reset grads for reference backward (q/k/v.grad no longer needed)
        q.grad = None
        k.grad = None
        v.grad = None

        # Dense reference backward: make fp32 leaves so both the eager forward
        # and its gradient reductions stay in fp32. The reference output and
        # resulting grads are cast back to the kernel dtype only for comparison.
        q_ref = q.detach().to(torch.float32).requires_grad_(True)
        k_ref = k.detach().to(torch.float32).requires_grad_(True)
        v_ref = v.detach().to(torch.float32).requires_grad_(True)
        expected_ref = dense_reference(
            q_ref, k_ref, v_ref, score_mod_name=score_mod_name,
            head_offset_buffer=head_offset_buffer,
            output_dtype=dtype,
        )
        torch.npu.synchronize()
        expected_ref.backward(grad_out)
        torch.npu.synchronize()
        assert_close_with_details(
            compiled_q_grad, q_ref.grad.to(dtype),
            tag=f"D {score_mod_name} B={B} Q={Q} KV={KV} q_grad {dtype}",
        )
        assert_close_with_details(
            compiled_k_grad, k_ref.grad.to(dtype),
            tag=f"D {score_mod_name} B={B} Q={Q} KV={KV} k_grad {dtype}",
        )
        assert_close_with_details(
            compiled_v_grad, v_ref.grad.to(dtype),
            tag=f"D {score_mod_name} B={B} Q={Q} KV={KV} v_grad {dtype}",
        )

    assert_frame_count(counter, f"D {score_mod_name}")
