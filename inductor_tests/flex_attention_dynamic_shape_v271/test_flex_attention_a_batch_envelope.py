"""Category A: dynamic batch with shape-specific BlockMasks.

Each runtime shape gets a broadcastable ``B=1, H=1`` BlockMask passed to one
compiled function. Runtime batch 1 is excluded because Dynamo 2.7 specializes
zero/one dimensions. The first runtime batch also differs from static ``H=2``
to avoid PyTorch 2.7 duck-shaping coupling it to Flex Attention's static head
dimension.

Run:
    pytest test_flex_attention_a_batch_envelope.py -v
"""
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    reset_dynamo_state,  # noqa: F401  (pytest fixture re-export)
    _causal_mask,
    _check_one_shape,
)


def _compile_flex(counter):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestBatchCapacityEnvelope:
    """A1-A4: dynamic input batches reuse broadcast BlockMask metadata."""

    def test_a1_basic_batch_dynamic(self, npu_device):
        """A1 baseline: runtime B=3 and B=4 reuse one graph."""
        counter = CompileCounterWithBackend("inductor")
        H, Q, KV, D = 2, 512, 512, 64

        compiled = _compile_flex(counter)
        for B in [3, 4]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"B={B}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across dynamic batches, got {counter.frame_count}"
        )

    def test_a2_batch_reuse_single_graph(self, npu_device):
        """A2: runtime B=3,4,8,16 reuse one dynamic graph."""
        counter = CompileCounterWithBackend("inductor")
        H, Q, KV, D = 2, 512, 512, 64

        compiled = _compile_flex(counter)
        for B in [3, 4, 8, 16]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"B={B}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile (runtime B=3..16), got {counter.frame_count}"
        )

    def test_a3_batch_broadcast_metadata(self, npu_device):
        """A3: runtime B=3,4,8 reuse broadcast mask metadata."""
        counter = CompileCounterWithBackend("inductor")
        H, Q, KV, D = 2, 256, 256, 64

        compiled = _compile_flex(counter)
        for B in [3, 4, 8]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"B={B} (broadcast)")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across runtime batches, got {counter.frame_count}"
        )

    def test_a4_batch_backward(self, npu_device):
        """A4: backward smoke checks across runtime B=3,4,8,16."""
        counter = CompileCounterWithBackend("inductor")
        H, Q, KV, D = 2, 256, 256, 64

        compiled = _compile_flex(counter)
        for B in [3, 4, 8, 16]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"B={B}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile (B dynamic + backward), got {counter.frame_count}"
        )
