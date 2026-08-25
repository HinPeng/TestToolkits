"""Category R: Q/KV crossing the 128 block-size boundary with shape-specific BlockMasks.

128 is the sparse block size; below-128 and above-128 are two logic paths on
NPU. Each runtime ``(Q, KV)`` shape gets an exact, broadcastable ``B=1, H=1``
BlockMask passed to one compiled function. The graph must be reused across
different Q/KV metadata capacities, including 128-boundary crossings.

fwd + dQ/dK/dV checked on every shape against the shared ``_check_one_shape``.

Run:
    pytest test_flex_attention_r_block_boundary_envelope.py -v
"""
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _causal_mask,
    _check_one_shape,
)

_B, _H, _D = 2, 2, 64   # static dims (B=2 to avoid Dynamo 0/1 specialization)


def _compile_flex(counter):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestBlockBoundaryEnvelope:
    """R: 128-block boundary crossings with exact dynamic-shape masks."""

    def test_r1_q_cross_128_same_capacity(self, npu_device):
        """R1: Q = 100 -> 127 -> 128 crosses 128 but stays at capacity 1.
        Exact masks per shape -> 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for q_len in [100, 127, 128]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=q_len, KV_LEN=256,
                                   device=npu_device)
            q = torch.randn(_B, _H, q_len, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(_B, _H, 256, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True,
                             tag=f"R1 q_len={q_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across the 128 boundary (same capacity), "
            f"got {counter.frame_count}"
        )

    def test_r2_q_capacity_ge2_dynamic(self, npu_device):
        """R2: Q = 129 -> 256 -> 257 -> 384, covering capacities 2 and 3.

        Capacity 1 is excluded because Dynamo specializes tensor dimensions
        of size 0/1 as a separate community-defined compile domain. Exact
        masks per shape -> 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for q_len in [129, 256, 257, 384]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=q_len, KV_LEN=256,
                                   device=npu_device)
            q = torch.randn(_B, _H, q_len, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(_B, _H, 256, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True,
                             tag=f"R2 q_len={q_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across Q capacities >= 2, "
            f"got {counter.frame_count}"
        )

    def test_r3_kv_from_below_to_above_128(self, npu_device):
        """R3: KV = 64 -> 127 -> 128 -> 129 (kv capacity 1 -> 2). Exact masks
        per shape -> 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for kv_len in [64, 127, 128, 129]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=256, KV_LEN=kv_len,
                                   device=npu_device)
            q = torch.randn(_B, _H, 256, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(_B, _H, kv_len, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False,
                             tag=f"R3 kv_len={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across KV 128 crossing, "
            f"got {counter.frame_count}"
        )

    def test_r4_q_below_kv_above(self, npu_device):
        """R4: below-128 Q with above-128 KV. Exact masks -> 1 compile."""
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for Q, KV in [(32, 128), (64, 512)]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV,
                                   device=npu_device)
            q = torch.randn(_B, _H, Q, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(_B, _H, KV, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False,
                             tag=f"R4 Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile (Q<128, KV>=128), "
            f"got {counter.frame_count}"
        )

    def test_r5_q_above_kv_below(self, npu_device):
        """R5: above-128 Q with below-128 KV (cross attention). Exact masks
        -> 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for Q, KV in [(128, 32), (512, 64)]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV,
                                   device=npu_device)
            q = torch.randn(_B, _H, Q, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(_B, _H, KV, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False,
                             tag=f"R5 Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile (Q>=128, KV<128), "
            f"got {counter.frame_count}"
        )

    def test_r6_both_below_128(self, npu_device):
        """R6: Q=KV both below 128 (decode-like tiny shapes). Exact masks
        -> 1 compile.
        """
        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for Q, KV in [(16, 16), (24, 32), (32, 32)]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV,
                                   device=npu_device)
            q = torch.randn(_B, _H, Q, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            k = torch.randn(_B, _H, KV, _D, device=npu_device, dtype=torch.bfloat16,
                            requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True,
                             tag=f"R6 Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile (both dims < 128), "
            f"got {counter.frame_count}"
        )
