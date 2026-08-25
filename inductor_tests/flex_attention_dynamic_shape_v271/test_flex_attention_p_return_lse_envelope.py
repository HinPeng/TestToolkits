"""Category P: return_lse=True with shape-specific BlockMasks.

Each runtime ``S`` shape gets an exact, broadcastable ``B=1, H=1`` BlockMask
passed to one compiled function. The graph must be reused across different S
metadata capacities.

Both output and LSE are checked per shape. The eager reference is computed
per shape with an EXACT-SIZED BlockMask.

Run:
    pytest test_flex_attention_p_return_lse_envelope.py -v
"""
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


def _compile_flex(counter, score_mod=None, return_lse=False):
    def attention(q, k, v, block_mask):
        return flex_attention(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            return_lse=return_lse,
        )

    return torch.compile(attention, backend=counter, dynamic=True)


class TestReturnLseEnvelope:
    """P: return_lse=True with exact dynamic-shape masks reusing one graph."""

    def test_lse_basic(self, npu_device):
        """P1: identity score_mod + noop_mask, exact masks over S = 256/384/512;
        out + LSE checked per shape. 1 compile.
        """
        B, H, D = 2, 8, 64
        dtype = torch.float32

        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(
            counter, score_mod=_base_identity_score_mod, return_lse=True
        )

        for S in [256, 384, 512]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
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
        """P2: causal mask + return_lse, exact masks over S = 256/384/512.

        BlockMask capacity starts at 2 so this test does not cross Dynamo's
        community-defined size-0/1 specialization domain. Out + LSE are
        checked per shape. 1 compile.
        """
        B, H, D = 2, 4, 64
        dtype = torch.float32

        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter, return_lse=True)

        for S in [256, 384, 512]:
            bm = create_block_mask(_base_causal_mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
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
