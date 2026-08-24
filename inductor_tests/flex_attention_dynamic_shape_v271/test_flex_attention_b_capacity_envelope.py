"""Category B: dynamic Q length with shape-specific BlockMasks.

Each Q length gets an exact BlockMask passed to one compiled function. Shapes
within the same Dynamo 2.7 capacity domain must reuse one graph. Capacity 1 is
tested separately because Dynamo specializes tensor dimensions whose size is
zero or one.

Run:
    pytest test_flex_attention_b_capacity_envelope.py -v
"""
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _causal_mask,
    _check_one_shape,
)


def _compile_flex(counter):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestQLenCapacityEnvelope:
    """B1-B5: exact Q-specific metadata capacities reuse one graph."""

    def test_b1_aligned_lengths_envelope(self, npu_device):
        """B1: exact masks at Q=KV=512,1024,2048 reuse one graph."""
        counter = CompileCounterWithBackend("inductor")
        H, D = 2, 64

        compiled = _compile_flex(counter)
        for q_len in [512, 1024, 2048]:
            bm = create_block_mask(_causal_mask, B=1, H=H, Q_LEN=q_len, KV_LEN=q_len, device=npu_device)
            q = torch.randn(1, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"Q=KV={q_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across aligned metadata capacities, got {counter.frame_count}"
        )

    def test_b2_unaligned_lengths_envelope(self, npu_device):
        """B2: unaligned Q lengths reuse a graph within each capacity domain."""
        H, KV, D = 2, 256, 64

        # Dynamo 2.7 specializes size 0/1, so capacity 1 and capacity >= 2 are
        # separate compile domains. Verify graph reuse within each domain.
        for q_lengths in ([3, 64, 100, 127], [129, 130, 257]):
            torch._dynamo.reset()
            counter = CompileCounterWithBackend("inductor")
            compiled = _compile_flex(counter)
            for q_len in q_lengths:
                bm = create_block_mask(_causal_mask, B=1, H=H, Q_LEN=q_len, KV_LEN=KV, device=npu_device)
                q = torch.randn(1, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
                k = torch.randn(1, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"Q={q_len}")

            assert counter.frame_count == 1, (
                f"Expected 1 compile within Q capacity domain {q_lengths}, "
                f"got {counter.frame_count}"
            )

    def test_b3_envelope_absorbs_capacity_boundary(self, npu_device):
        """B3: exact masks across the 256 -> 257 capacity boundary reuse one graph."""
        counter = CompileCounterWithBackend("inductor")
        H, KV, D = 2, 256, 64

        compiled = _compile_flex(counter)
        for q_len in [256, 257]:
            bm = create_block_mask(_causal_mask, B=1, H=H, Q_LEN=q_len, KV_LEN=KV, device=npu_device)
            q = torch.randn(1, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"Q={q_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across Q metadata capacity 2 -> 3, "
            f"got {counter.frame_count}"
        )

    def test_b4_backward_envelope(self, npu_device):
        """B4: backward smoke checks across exact Q=129,257,512 masks."""
        counter = CompileCounterWithBackend("inductor")
        H, KV, D = 2, 256, 64

        compiled = _compile_flex(counter)
        for q_len in [129, 257, 512]:
            bm = create_block_mask(_causal_mask, B=1, H=H, Q_LEN=q_len, KV_LEN=KV, device=npu_device)
            q = torch.randn(1, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"Q={q_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across backward Q capacities, got {counter.frame_count}"
        )

    def test_b5_q_reuse_backward_kv_fixed(self, npu_device):
        """B5: exact Q=512,1024,2048 masks with fixed KV reuse one graph."""
        counter = CompileCounterWithBackend("inductor")
        B, H, KV, D = 1, 2, 512, 64

        compiled = _compile_flex(counter)
        for q_len in [512, 1024, 2048]:
            bm = create_block_mask(_causal_mask, B=B, H=H, Q_LEN=q_len, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"Q={q_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across large Q metadata capacities, "
            f"got {counter.frame_count}"
        )
