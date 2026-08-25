"""Category S: large Q/KV lengths (4K -> 8K -> 16K) with shape-specific BlockMasks.

Each runtime ``S`` (or ``(Q, KV)``) shape gets an exact, broadcastable
``B=1, H=1`` BlockMask passed to one compiled function. The graph must be
reused across different S metadata capacities.

Reference is the dense SDPA math (``_dense_reference``) for parity with
A/B/C/D: avoids comparing flex_attention against itself, which would be
circular. At 4K-16K the [S,S] dense score matrix fits comfortably in NPU
memory. Forward only.

Run:
    pytest test_flex_attention_s_large_length_envelope.py -v
"""
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    assert_close_with_details,
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _dense_reference,
    FWD_ATOL,
    FWD_RTOL,
)

_B, _H, _D = 2, 2, 64   # static dims (B=2 to avoid Dynamo 0/1 specialization)


def _compile_flex(counter):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


def _check_large_shape(compiled, q, k, v, bm, tag):
    """Compiled output vs dense SDPA reference (no flex_attention self-compare)."""
    expected = _dense_reference(q, k, v, causal=False, score_fn=None)
    actual = compiled(q, k, v, bm)
    assert_close_with_details(actual, expected, atol=FWD_ATOL, rtol=FWD_RTOL, msg=tag)


class TestLargeLengthEnvelope:
    """S: 4K -> 8K -> 16K lengths with exact dynamic-shape masks, fwd only."""

    def test_s1_large_square_envelope(self, npu_device):
        """S1: square shapes 4K -> 8K -> 16K with exact masks per shape.
        Dense reference per shape. Assert 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for seq_len in [4096, 8192, 16384]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len,
                                   device=npu_device)
            q = torch.randn(_B, _H, seq_len, _D, device=npu_device, dtype=torch.bfloat16)
            k = torch.randn(_B, _H, seq_len, _D, device=npu_device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            _check_large_shape(compiled, q, k, v, bm,
                               tag=f"square fwd mismatch at seq_len={seq_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across 4K/8K/16K exact masks, "
            f"got {counter.frame_count}"
        )

    def test_s2_asymmetric_large_envelope(self, npu_device):
        """S2: asymmetric large cross attention with exact masks per shape.
        Runtime (Q,KV) = 4K/4K -> 4K/8K -> 8K/8K.
        Dense reference per shape. Assert 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for Q, KV in [(4096, 4096), (4096, 8192), (8192, 8192)]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV,
                                   device=npu_device)
            q = torch.randn(_B, _H, Q, _D, device=npu_device, dtype=torch.bfloat16)
            k = torch.randn(_B, _H, KV, _D, device=npu_device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            _check_large_shape(compiled, q, k, v, bm,
                               tag=f"asymmetric fwd mismatch at Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across asymmetric large exact masks, "
            f"got {counter.frame_count}"
        )
