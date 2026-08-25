"""Category N: remainder (%) operators with shape-specific BlockMasks.

The dynamic operator tests use one compiled function across runtime ``S``
shapes. Runtime and eager-reference BlockMasks use the same real ``B``/``H``
dimensions so head-dependent masks have identical semantics.

The eager reference is computed per shape with an EXACT-SIZED BlockMask.

N1/N4/N4b keep their original single-shape structure and verify supported
remainder forms directly.

Run:
    pytest test_flex_attention_n_remainder_envelope.py -v
"""
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    assert_close_with_details,
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _base_causal_mask_mod,
    _remainder_mask_mod,
    _remainder_mod3_mask,
    _remainder_boolean_mask,
    _remainder_in_score_mod,
    _bitwise_and_mask,
    _equiv_form_mask,
    _floordiv_mask,
)

_B, _H, _D = 2, 8, 64
_RUNTIME_S = [256, 384, 512]


def _compile_flex(counter, score_mod=None):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


def _run_mask_envelope(npu_device, mask_mod, *, score_mod=None, with_backward=False):
    """One graph over _RUNTIME_S with exact masks, per-shape eager reference,
    asserts frame_count == 1."""
    counter = CompileCounterWithBackend("inductor")
    compiled = _compile_flex(counter, score_mod)

    for S in _RUNTIME_S:
        bm = create_block_mask(mask_mod, B=_B, H=_H, Q_LEN=S, KV_LEN=S,
                               device=npu_device)
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

        assert_close_with_details(
            comp_out, ref_out, atol=1e-2, rtol=1e-2, msg=f"fwd mismatch @ S={S}"
        )
        if with_backward:
            assert_close_with_details(
                q.grad,
                ref_q_grad,
                atol=1e-2,
                rtol=1e-2,
                msg=f"q.grad mismatch @ S={S}",
            )
            q.grad = k.grad = v.grad = None

    assert counter.frame_count == 1, (
        f"Expected 1 compile across exact masks, got {counter.frame_count}"
    )


class TestRemainderEnvelope:
    """N: remainder (%) operator guards with exact dynamic-shape masks."""

    def test_remainder_in_mask_mod_passes(self, npu_device):
        """N1: mask_mod with % works on NPU Inductor (single shape)."""
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
        assert_close_with_details(
            comp_out, ref_out, atol=1e-2, rtol=1e-2,
            msg="remainder in mask_mod @ S=256",
        )

    def test_remainder_in_score_mod_passes(self, npu_device):
        """N2: % in score_mod works fine, exact masks, fwd+bwd."""
        _run_mask_envelope(npu_device, _base_causal_mask_mod,
                           score_mod=_remainder_in_score_mod, with_backward=True)

    def test_bitwise_and_in_mask_mod_passes(self, npu_device):
        """N3: & 1 works in mask_mod, exact masks."""
        _run_mask_envelope(npu_device, _bitwise_and_mask)

    def test_remainder_diff_divisor(self, npu_device):
        """N4: % 3 works in mask_mod (single shape)."""
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
        assert_close_with_details(
            comp_out, ref_out, atol=1e-2, rtol=1e-2,
            msg="remainder divisor=3 @ S=256",
        )

    def test_remainder_boolean_comparison(self, npu_device):
        """N4b: % with an explicit boolean comparison works (single shape)."""
        B, H, S, D = 2, 8, 256, 64
        q = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        bm = create_block_mask(_remainder_boolean_mask, B=B, H=H, Q_LEN=S, KV_LEN=S, device=npu_device)

        def op_fn(q_, k_, v_):
            return flex_attention(q_, k_, v_, block_mask=bm)

        ref_out = op_fn(q, k, v)
        compiled_fn = torch.compile(op_fn, backend="inductor", dynamic=True)
        comp_out = compiled_fn(q, k, v)
        assert_close_with_details(
            comp_out, ref_out, atol=1e-2, rtol=1e-2,
            msg="bitwise_and in mask_mod @ S=256",
        )

    def test_remainder_equivalent_form(self, npu_device):
        """N5: h - (h // 2) * 2 works (no % operator), exact masks."""
        _run_mask_envelope(npu_device, _equiv_form_mask)

    def test_floordiv_in_mask_mod_passes(self, npu_device):
        """N6: // (floordiv) works in mask_mod, exact masks."""
        _run_mask_envelope(npu_device, _floordiv_mask)
