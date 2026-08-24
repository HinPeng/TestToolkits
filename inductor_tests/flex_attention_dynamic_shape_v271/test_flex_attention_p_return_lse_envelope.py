"""Category P: return_lse=True with shape-specific BlockMasks.

Each runtime ``S`` shape gets an exact, broadcastable ``B=1, H=1`` BlockMask
captured by ``functools.partial`` (together with ``score_mod`` and
``return_lse``). All compiled partials share one backend and must reuse one
graph across different S metadata capacities, matching community M01.

Both output and LSE are checked per shape. The eager reference is computed
per shape with an EXACT-SIZED BlockMask.

Run:
    pytest test_flex_attention_p_return_lse_envelope.py -v
"""
import functools

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _base_identity_score_mod,
    _base_causal_mask_mod,
)


def _compile_flex_with_mask(counter, block_mask, score_mod=None, return_lse=False):
    attention = functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        return_lse=return_lse,
    )
    compiled = torch.compile(attention, backend=counter, dynamic=True)

    def call(q, k, v, _unused_block_mask):
        return compiled(q, k, v)

    return call


class TestReturnLseEnvelope:
    """P: return_lse=True with exact dynamic-shape masks reusing one graph."""

    def test_lse_basic(self, npu_device):
        """P1: identity score_mod + noop_mask, exact masks over S = 256/384/512;
        out + LSE checked per shape. 1 compile.
        """
        B, H, D = 2, 8, 64
        dtype = torch.float32

        counter = CompileCounterWithBackend("inductor")

        for S in [256, 384, 512]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
            compiled = _compile_flex_with_mask(counter, bm,
                                                score_mod=_base_identity_score_mod,
                                                return_lse=True)
            q = torch.randn(B, H, S, D, device=npu_device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=npu_device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=npu_device, dtype=dtype)

            # Eager reference with an exact-sized mask for this shape
            bm_ref = create_block_mask(noop_mask, B=B, H=H, Q_LEN=S, KV_LEN=S,
                                       device=npu_device)
            ref_out, ref_lse = flex_attention(q, k, v, score_mod=_base_identity_score_mod,
                                              block_mask=bm_ref, return_lse=True)

            comp_out, comp_lse = compiled(q, k, v, bm)

            out_diff = torch.max(torch.abs(ref_out - comp_out)).item()
            assert out_diff < 1e-2, f"Output mismatch @ S={S}: max_diff={out_diff}"

            lse_aligned = torch.allclose(ref_lse, comp_lse, atol=0.01, rtol=0.01)
            assert lse_aligned, (
                f"LSE offset bug @ S={S}: "
                f"mean_diff={(ref_lse - comp_lse).mean().item():.4f}, "
                f"std_diff={(ref_lse - comp_lse).std().item():.4f}"
            )

        assert counter.frame_count == 1, (
            f"Expected 1 compile across exact masks, got {counter.frame_count}"
        )

    def test_lse_causal(self, npu_device):
        """P2: causal mask + return_lse, exact masks over S = 128/256;
        out + LSE checked per shape. 1 compile.
        """
        B, H, D = 2, 4, 64
        dtype = torch.float32

        counter = CompileCounterWithBackend("inductor")

        for S in [128, 256]:
            bm = create_block_mask(_base_causal_mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
            compiled = _compile_flex_with_mask(counter, bm, return_lse=True)
            q = torch.randn(B, H, S, D, device=npu_device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=npu_device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=npu_device, dtype=dtype)

            bm_ref = create_block_mask(_base_causal_mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=S,
                                       device=npu_device)
            ref_out, ref_lse = flex_attention(q, k, v, block_mask=bm_ref, return_lse=True)

            comp_out, comp_lse = compiled(q, k, v, bm)

            out_diff = torch.max(torch.abs(ref_out - comp_out)).item()
            assert out_diff < 1e-2, f"Output mismatch @ S={S}: max_diff={out_diff}"

            lse_aligned = torch.allclose(ref_lse, comp_lse, atol=0.01, rtol=0.01)
            assert lse_aligned, f"LSE mismatch under causal mask @ S={S}"

        assert counter.frame_count == 1, (
            f"Expected 1 compile across exact causal masks, got {counter.frame_count}"
        )
