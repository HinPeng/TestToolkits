"""Category N: remainder (%) operator guards with shape-specific BlockMasks.

The passing operator tests (N2/N3/N5/N6) use the same fix pattern as A-D:
each runtime ``S`` shape gets an exact, broadcastable ``B=1, H=1`` BlockMask
captured by ``functools.partial``. All compiled partials share one backend
and must reuse one graph across different S metadata capacities.

The eager reference is computed per shape with an EXACT-SIZED BlockMask.

The three xfail guard tests (N1/N4/N4b: aten.remainder.Scalar unsupported in
mask_mod lowering on NPU) keep their original single-shape structure: they
are expected to fail on the very first compiled call.

Run:
    pytest test_flex_attention_n_remainder_envelope.py -v
"""
import functools

import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _base_causal_mask_mod,
    _remainder_mask_mod,
    _remainder_mod3_mask,
    _remainder_no_compare_mask,
    _remainder_in_score_mod,
    _bitwise_and_mask,
    _equiv_form_mask,
    _floordiv_mask,
)

_B, _H, _D = 2, 8, 64
_RUNTIME_S = [256, 384, 512]


def _compile_flex_with_mask(counter, block_mask, score_mod=None):
    attention = functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
    )
    compiled = torch.compile(attention, backend=counter, dynamic=True)

    def call(q, k, v, _unused_block_mask):
        return compiled(q, k, v)

    return call


def _run_mask_envelope(npu_device, mask_mod, *, score_mod=None, with_backward=False):
    """One graph over _RUNTIME_S with exact masks, per-shape eager reference,
    asserts frame_count == 1."""
    counter = CompileCounterWithBackend("inductor")

    for S in _RUNTIME_S:
        bm = create_block_mask(mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S,
                               device=npu_device)
        compiled = _compile_flex_with_mask(counter, bm, score_mod)
        q = torch.randn(_B, _H, S, _D, device=npu_device, dtype=torch.float32,
                        requires_grad=True)
        k = torch.randn(_B, _H, S, _D, device=npu_device, dtype=torch.float32,
                        requires_grad=True)
        v = torch.randn(_B, _H, S, _D, device=npu_device, dtype=torch.float32,
                        requires_grad=True)

        # Eager reference with an exact-sized mask for this shape
        bm_ref = create_block_mask(mask_mod, B=_B, H=_H, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
        ref_out = flex_attention(q, k, v, score_mod=score_mod, block_mask=bm_ref)
        if with_backward:
            grad_out = torch.randn_like(ref_out)
            ref_out.backward(grad_out)
            ref_q_grad = q.grad.clone()
            q.grad = k.grad = v.grad = None
        else:
            grad_out = None

        comp_out = compiled(q, k, v, bm)
        if with_backward:
            comp_out.backward(grad_out)

        torch.testing.assert_close(comp_out, ref_out, atol=1e-2, rtol=1e-2,
                                   msg=f"fwd mismatch @ S={S}")
        if with_backward:
            torch.testing.assert_close(q.grad, ref_q_grad, atol=1e-2, rtol=1e-2,
                                       msg=f"q.grad mismatch @ S={S}")
            q.grad = k.grad = v.grad = None

    assert counter.frame_count == 1, (
        f"Expected 1 compile across exact masks, got {counter.frame_count}"
    )


class TestRemainderEnvelope:
    """N: remainder (%) operator guards with exact dynamic-shape masks."""

    @pytest.mark.xfail(
        reason="NPU bug: aten.remainder.Scalar not supported in mask_mod subgraph lowering",
        strict=True,
    )
    def test_remainder_in_mask_mod_fails(self, npu_device):
        """N1: mask_mod with % should fail on NPU Inductor (single shape)."""
        B, H, S, D = 2, 8, 256, 64
        q = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        bm = create_block_mask(_remainder_mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=S, device=npu_device)

        def op_fn(q_, k_, v_):
            return flex_attention(q_, k_, v_, block_mask=bm)

        ref_out = op_fn(q, k, v)
        compiled_fn = torch.compile(op_fn, backend="inductor", dynamic=True)
        comp_out = compiled_fn(q, k, v)
        torch.testing.assert_close(comp_out, ref_out, atol=1e-2, rtol=1e-2)

    def test_remainder_in_score_mod_passes(self, npu_device):
        """N2: % in score_mod works fine, exact masks, fwd+bwd."""
        _run_mask_envelope(npu_device, _base_causal_mask_mod,
                           score_mod=_remainder_in_score_mod, with_backward=True)

    def test_bitwise_and_in_mask_mod_passes(self, npu_device):
        """N3: & 1 works in mask_mod, exact masks."""
        _run_mask_envelope(npu_device, _bitwise_and_mask)

    @pytest.mark.xfail(
        reason="NPU bug: aten.remainder.Scalar with any divisor fails in mask_mod",
        strict=True,
    )
    def test_remainder_diff_divisor(self, npu_device):
        """N4: % 3 also fails (single shape)."""
        B, H, S, D = 2, 8, 256, 64
        q = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        bm = create_block_mask(_remainder_mod3_mask, B=B, H=H, Q_LEN=S, KV_LEN=S, device=npu_device)

        def op_fn(q_, k_, v_):
            return flex_attention(q_, k_, v_, block_mask=bm)

        ref_out = op_fn(q, k, v)
        compiled_fn = torch.compile(op_fn, backend="inductor", dynamic=True)
        comp_out = compiled_fn(q, k, v)
        torch.testing.assert_close(comp_out, ref_out, atol=1e-2, rtol=1e-2)

    @pytest.mark.xfail(
        reason="NPU bug: aten.remainder.Scalar in any form fails in mask_mod",
        strict=True,
    )
    def test_remainder_with_comparison_only(self, npu_device):
        """N4b: % without == (h % 2 as boolean) still fails (single shape)."""
        B, H, S, D = 2, 8, 256, 64
        q = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        bm = create_block_mask(_remainder_no_compare_mask, B=B, H=H, Q_LEN=S, KV_LEN=S, device=npu_device)

        def op_fn(q_, k_, v_):
            return flex_attention(q_, k_, v_, block_mask=bm)

        ref_out = op_fn(q, k, v)
        compiled_fn = torch.compile(op_fn, backend="inductor", dynamic=True)
        comp_out = compiled_fn(q, k, v)
        torch.testing.assert_close(comp_out, ref_out, atol=1e-2, rtol=1e-2)

    def test_remainder_equivalent_form(self, npu_device):
        """N5: h - (h // 2) * 2 works (no % operator), exact masks."""
        _run_mask_envelope(npu_device, _equiv_form_mask)

    def test_floordiv_in_mask_mod_passes(self, npu_device):
        """N6: // (floordiv) works in mask_mod, exact masks."""
        _run_mask_envelope(npu_device, _floordiv_mask)
