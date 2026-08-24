"""Category J: GQA (grouped query attention) with shape-specific BlockMasks.

Each runtime ``(B, Q, KV)`` shape gets an exact, broadcastable ``B=1, H=1``
BlockMask captured by ``functools.partial`` (together with
``enable_gqa=True``). All compiled partials share one backend and must reuse
one graph across different B/Q/KV metadata capacities, matching community M01.

H_Q / H_KV / head_dim are static dims and stay fixed; GQA is exercised via
``enable_gqa=True`` on ``flex_attention``. fwd + dQ verified on every shape
against the dense reference with repeat-expanded KV heads (same reference
usage as the original J1).

Per-shape tolerances use local constants because the shared
``_check_one_shape`` uses a single 8e-2 tolerance and does not exercise GQA.

Run:
    pytest test_flex_attention_j_gqa_envelope.py -v
"""
import functools

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _causal_mask,
    _dense_reference,
)

_H_Q, _H_KV, _D = 16, 8, 64     # static dims: query/kv heads, head_dim

FWD_ATOL = 2e-2
FWD_RTOL = 2e-2
GRAD_ATOL = 8e-2
GRAD_RTOL = 8e-2


def _compile_flex_with_mask(counter, block_mask):
    attention = functools.partial(flex_attention, block_mask=block_mask, enable_gqa=True)
    compiled = torch.compile(attention, backend=counter, dynamic=True)

    def call(q, k, v, _unused_block_mask):
        return compiled(q, k, v)

    return call


def _check_gqa_shape(compiled, q, k, v, bm, tag):
    """fwd + dQ vs dense reference with repeat-expanded KV heads."""
    # Reference: repeat KV heads to match Q heads
    k_expanded = k.detach().repeat_interleave(_H_Q // _H_KV, dim=1).requires_grad_(True)
    v_expanded = v.detach().repeat_interleave(_H_Q // _H_KV, dim=1).requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    expected = _dense_reference(q_ref, k_expanded, v_expanded, causal=True, score_fn=None)
    expected_grads = torch.autograd.grad(expected.sum(), (q_ref, k_expanded, v_expanded))

    actual = compiled(q, k, v, bm)
    actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))

    torch.testing.assert_close(actual, expected, atol=FWD_ATOL, rtol=FWD_RTOL, msg=tag)
    # q grad direct comparison (same usage as the original J1)
    torch.testing.assert_close(actual_grads[0], expected_grads[0],
                               atol=GRAD_ATOL, rtol=GRAD_RTOL, msg=tag)


class TestGQAEnvelope:
    """J: GQA 16q/8kv heads, exact masks for varying (B, Q, KV) reuse one graph."""

    def test_j1_gqa_dynamic(self, npu_device):
        """Exact BlockMask per (B, Q, KV) shape, captured via partial with
        ``enable_gqa=True``; all compiled partials share one backend
        -> 1 compile. Runtime shapes span Q != KV; fwd+dQ checked per shape.
        """
        counter = CompileCounterWithBackend("inductor")

        for B, Q, KV in [(2, 512, 512), (2, 512, 1024), (4, 1024, 2048)]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV,
                                   device=npu_device)
            compiled = _compile_flex_with_mask(counter, bm)
            q = torch.randn(B, _H_Q, Q, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(B, _H_KV, KV, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_gqa_shape(compiled, q, k, v, bm, tag=f"GQA B={B},Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across exact GQA masks, "
            f"got {counter.frame_count}"
        )
