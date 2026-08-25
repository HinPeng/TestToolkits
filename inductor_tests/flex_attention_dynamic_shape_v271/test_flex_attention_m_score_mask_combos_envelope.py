"""Category M: extended score_mod x mask_mod combos with shape-specific BlockMasks.

Each runtime ``S`` shape gets an exact ``B=2, H=8`` BlockMask passed to one
compiled function. Runtime and eager-reference BlockMasks use identical
batch/head semantics, including for head-dependent masks. The graph must be
reused across different S metadata capacities.

The eager reference is computed per shape with an EXACT-SIZED BlockMask, so
the reference side never relies on envelope reuse.

Expected-failure combos (fully masked rows -> NaN) keep the original
single-shape structure: the test is expected to fail on the very first
compiled call.

Run:
    pytest test_flex_attention_m_score_mask_combos_envelope.py -v
"""
import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _COMBO_SCORE_MODS,
    _COMBO_MASK_MODS,
    _VALID_COMBOS,
    _EXPECTED_FAILURE_COMBOS,
)

_B, _H, _D = 2, 8, 64
_RUNTIME_S = [256, 384, 512]


def _compile_flex(counter, score_mod=None):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestScoreMaskCombosEnvelope:
    """M: score_mod x mask_mod combos, exact masks reuse one graph per combo."""

    @pytest.mark.parametrize(
        "score_mod_name,mask_mod_name",
        _VALID_COMBOS,
        ids=[f"{s}--x--{m}" for s, m in _VALID_COMBOS],
    )
    def test_combo(self, npu_device, score_mod_name, mask_mod_name):
        """One combo, exact masks over S = 256/384/512, fwd+bwd each."""
        score_mod = _COMBO_SCORE_MODS[score_mod_name]
        mask_mod = _COMBO_MASK_MODS[mask_mod_name]

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
            grad_out = torch.randn_like(ref_out)
            ref_out.backward(grad_out)
            ref_q_grad = q.grad.clone()
            ref_k_grad = k.grad.clone()
            ref_v_grad = v.grad.clone()
            q.grad = k.grad = v.grad = None

            comp_out = compiled(q, k, v, bm)
            comp_out.backward(grad_out)

            atol, rtol = 1e-2, 1e-2
            tag = f"{score_mod_name} x {mask_mod_name} @ S={S}"
            torch.testing.assert_close(comp_out, ref_out, atol=atol, rtol=rtol,
                                       msg=f"Fwd mismatch: {tag}")
            torch.testing.assert_close(q.grad, ref_q_grad, atol=atol, rtol=rtol,
                                       msg=f"q.grad mismatch: {tag}")
            torch.testing.assert_close(k.grad, ref_k_grad, atol=atol, rtol=rtol,
                                       msg=f"k.grad mismatch: {tag}")
            torch.testing.assert_close(v.grad, ref_v_grad, atol=atol, rtol=rtol,
                                       msg=f"v.grad mismatch: {tag}")
            q.grad = k.grad = v.grad = None

        assert counter.frame_count == 1, (
            f"{score_mod_name} x {mask_mod_name}: expected 1 compile across exact masks, "
            f"got {counter.frame_count}"
        )

    @pytest.mark.parametrize(
        "score_mod_name,mask_mod_name",
        list(_EXPECTED_FAILURE_COMBOS),
        ids=[f"{s}-x-{m}" for s, m in _EXPECTED_FAILURE_COMBOS],
    )
    @pytest.mark.xfail(reason="Fully masked rows produce NaN, expected behavior")
    def test_expected_failure_combo(self, npu_device, score_mod_name, mask_mod_name):
        """Expected failure combos (fully masked -> NaN), single shape."""
        score_mod = _COMBO_SCORE_MODS[score_mod_name]
        mask_mod = _COMBO_MASK_MODS[mask_mod_name]
        B, H, S, D = 2, 8, 256, 64

        q = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, S, D, device=npu_device, dtype=torch.float32, requires_grad=True)

        bm = create_block_mask(mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=S, device=npu_device)

        def op_fn(q_, k_, v_):
            return flex_attention(q_, k_, v_, score_mod=score_mod, block_mask=bm)

        ref_out = op_fn(q, k, v)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        ref_q_grad = q.grad.clone()
        q.grad = k.grad = v.grad = None

        compiled_fn = torch.compile(op_fn, backend="inductor", dynamic=True)
        comp_out = compiled_fn(q, k, v)
        comp_out.backward(grad_out)

        # These assertions are expected to fail (NaN)
        torch.testing.assert_close(comp_out, ref_out, atol=1e-2, rtol=1e-2)
