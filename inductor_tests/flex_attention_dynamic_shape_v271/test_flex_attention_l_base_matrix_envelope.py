"""Category L: base matrix with shape-specific BlockMasks.

score_mod (identity/causal/rel_bias) x mask_mod (noop/causal) x dtype
(fp16/fp32/bf16) = 18 combos, forward + backward.

Each runtime ``(B, S)`` shape gets an exact, broadcastable ``B=1, H=1``
BlockMask passed to one compiled function. The eager reference is computed
per shape with an exact-sized BlockMask;
fwd + dQ/dK/dV are verified against it. No ``frame_count`` assertion is
made here (per-shape ``compiled`` + counter interaction is unreliable on
NPU), matching A/B/C/D's "numerical-only" style.

Run:
    pytest test_flex_attention_l_base_matrix_envelope.py -v
"""
import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _BASE_SCORE_MODS,
    _BASE_MASK_MODS,
    _BASE_DTYPES,
)


def _compile_flex(counter, score_mod=None):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestBaseMatrixEnvelope:
    """L: 18 base combos with exact dynamic-shape masks reusing one graph.

    H/head_dim are static dims (fixed per family); only B/S vary at runtime.
    H=4 and H=8 families are separate graphs by design (H is static).
    """

    # Shape families: only B/S vary, H/D fixed (base shapes retained as first entry).
    _L_FAMILY_H4 = [(2, 4, 128, 64), (4, 4, 256, 64), (2, 4, 512, 64)]
    _L_FAMILY_H8 = [(2, 8, 256, 64), (4, 8, 512, 64)]

    @pytest.mark.parametrize("score_mod_name", list(_BASE_SCORE_MODS.keys()))
    @pytest.mark.parametrize("mask_mod_name", list(_BASE_MASK_MODS.keys()))
    @pytest.mark.parametrize("dtype", _BASE_DTYPES, ids=["fp16", "fp32", "bf16"])
    def test_base_matrix(self, npu_device, score_mod_name, mask_mod_name, dtype):
        """Base matrix: score x mask x dtype with exact dynamic-shape masks.

        fwd + dQ/dK/dV verified against an eager reference computed with an
        exact-sized BlockMask per shape. Tolerances match the base test (1e-1).
        """
        score_mod = _BASE_SCORE_MODS[score_mod_name]
        mask_mod = _BASE_MASK_MODS[mask_mod_name]
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter, score_mod)

        for (B, H, S, D) in self._L_FAMILY_H4 + self._L_FAMILY_H8:
            bm = create_block_mask(mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
            q = torch.randn(B, H, S, D, device=npu_device, dtype=dtype, requires_grad=True)
            k = torch.randn(B, H, S, D, device=npu_device, dtype=dtype, requires_grad=True)
            v = torch.randn(B, H, S, D, device=npu_device, dtype=dtype, requires_grad=True)

            # Eager reference with an exact-sized mask for this shape
            bm_ref = create_block_mask(mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=S,
                                       device=npu_device)
            eager_out = flex_attention(q, k, v, score_mod=score_mod, block_mask=bm_ref)
            grad_out = torch.randn_like(eager_out)
            eager_out.backward(grad_out)
            eager_q_grad = q.grad.clone()
            eager_k_grad = k.grad.clone()
            eager_v_grad = v.grad.clone()
            q.grad = k.grad = v.grad = None

            compiled_out = compiled(q, k, v, bm)
            compiled_out.backward(grad_out)

            # Use same loose tolerance as base test (1e-1)
            atol, rtol = 1e-1, 0.0
            shape_tag = f"{B}x{H}x{S}x{D}"
            torch.testing.assert_close(
                compiled_out, eager_out, atol=atol, rtol=rtol,
                msg=f"fwd mismatch {score_mod_name}/{mask_mod_name}/{dtype} @ {shape_tag}")
            torch.testing.assert_close(q.grad, eager_q_grad, atol=atol, rtol=rtol)
            torch.testing.assert_close(k.grad, eager_k_grad, atol=atol, rtol=rtol)
            torch.testing.assert_close(v.grad, eager_v_grad, atol=atol, rtol=rtol)
            q.grad = k.grad = v.grad = None
