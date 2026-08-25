"""Category I: dtype x dynamic shape with shape-specific BlockMasks.

Each runtime ``(Q, KV)`` shape gets an exact, broadcastable ``B=1, H=1``
BlockMask passed to one compiled function. The graph must be reused across
different Q/KV metadata capacities.

dtype stays parameterized: each dtype is its own graph (dtype is a static
specialization dimension, not a runtime dynamic dim). Per-dtype tolerances
are kept via a local ``_check_dtype_shape`` helper because the shared
``_check_one_shape`` uses a single 8e-2 tolerance.

Run:
    pytest test_flex_attention_i_dtype_envelope.py -v
"""
import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    assert_close_with_details,
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _causal_mask,
    _dense_reference,
)


_B, _H, _D = 1, 2, 64                 # static dims; B=1 because dtype is the
                                       # specialization axis under test, not batch
_RUNTIME_SHAPES = [(256, 256), (256, 512), (512, 512)]


def _compile_flex(counter):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


def _check_dtype_shape(compiled, q, k, v, bm, *, atol, rtol, tag):
    """fwd + dQ/dK/dV vs dense reference with per-dtype tolerances."""
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    expected = _dense_reference(q_ref, k_ref, v_ref, causal=True, score_fn=None)
    expected_grads = torch.autograd.grad(expected.sum(), (q_ref, k_ref, v_ref))

    actual = compiled(q, k, v, bm)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))

    assert_close_with_details(actual, expected, atol=atol, rtol=rtol, msg=tag)
    for i, name in enumerate(("q", "k", "v")):
        assert_close_with_details(
            actual_grads[i], expected_grads[i],
            atol=atol * 4, rtol=rtol * 4,
            msg=f"{name}.grad mismatch at {tag}",
        )


class TestDtypeEnvelope:
    """I: fp32/bf16/fp16 with exact dynamic-shape masks reusing one graph."""

    @pytest.mark.parametrize("dtype,atol,rtol", [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float16, 5e-3, 5e-3),
    ])
    def test_i1_dtype_dynamic_shapes(self, npu_device, dtype, atol, rtol):
        """Exact BlockMask per (Q, KV) shape passed at runtime to one compiled
        function -> 1 compile per dtype.
        fwd+bwd numerically checked at per-dtype tolerance on every shape.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for Q, KV in _RUNTIME_SHAPES:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV,
                                   device=npu_device)
            q = torch.randn(_B, _H, Q, _D, device=npu_device, dtype=dtype,
                            requires_grad=True)
            k = torch.randn(_B, _H, KV, _D, device=npu_device, dtype=dtype,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_dtype_shape(compiled, q, k, v, bm, atol=atol, rtol=rtol,
                               tag=f"dtype={dtype}, Q={Q}, KV={KV}")

        assert counter.frame_count == 1, (
            f"dtype={dtype}: expected 1 compile across exact masks, "
            f"got {counter.frame_count}"
        )
