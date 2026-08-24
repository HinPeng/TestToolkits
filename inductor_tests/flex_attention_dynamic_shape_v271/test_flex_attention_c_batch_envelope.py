"""Category C: dynamic KV length and Q != KV with shape-specific BlockMasks.

Each (Q, KV) pair gets an exact BlockMask passed to one compiled function.
The function must reuse one graph across dynamic metadata capacities greater
than one; Dynamo 2.7's zero/one dimension specialization is kept out of these
tests.

Run:
    pytest test_flex_attention_c_batch_envelope.py -v
"""
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    noop_mask,
)
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


class TestKVLenCapacityEnvelope:
    """C1-C7: exact (Q, KV)-specific metadata capacities reuse one graph."""

    def test_c1_kv_variation_same_and_cross_capacity(self, npu_device):
        """C1: exact masks for Q=512 and KV=900,1024,2048 reuse one graph."""
        counter = CompileCounterWithBackend("inductor")
        B, H, Q, D = 1, 2, 512, 64

        compiled = _compile_flex(counter)
        for kv_len in [900, 1024, 2048]:
            bm = create_block_mask(noop_mask, B=B, H=H, Q_LEN=Q, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"Q={Q},KV={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across KV metadata capacities, got {counter.frame_count}"
        )

    def test_c2_q_less_than_kv(self, npu_device):
        """C2: exact cross-attention masks for Q=256 < dynamic KV."""
        counter = CompileCounterWithBackend("inductor")
        B, H, Q, D = 1, 2, 256, 64

        compiled = _compile_flex(counter)
        for kv_len in [512, 1024, 4096]:
            bm = create_block_mask(noop_mask, B=B, H=H, Q_LEN=Q, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"Q={Q}<KV={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile with Q<KV metadata changes, got {counter.frame_count}"
        )

    def test_c3_q_greater_than_kv(self, npu_device):
        """C3: exact masks while Q > KV and both lengths vary."""
        counter = CompileCounterWithBackend("inductor")
        B, H, D = 1, 2, 64

        compiled = _compile_flex(counter)
        for q_len, kv_len in [(1024, 256), (2048, 129), (2048, 192)]:
            bm = create_block_mask(noop_mask, B=B, H=H, Q_LEN=q_len, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"Q={q_len}>KV={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile (Q>KV with exact masks), "
            f"got {counter.frame_count}"
        )

    def test_c4_kv_unaligned(self, npu_device):
        """C4: exact masks for non-aligned KV tail blocks."""
        counter = CompileCounterWithBackend("inductor")
        B, H, Q, D = 1, 2, 256, 64

        compiled = _compile_flex(counter)
        for kv_len in [129, 257, 300]:
            bm = create_block_mask(noop_mask, B=B, H=H, Q_LEN=Q, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"Q={Q},KV={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across unaligned KV capacities, got {counter.frame_count}"
        )

    def test_c5_both_unaligned(self, npu_device):
        """C5: exact causal masks with Q and KV both non-aligned."""
        counter = CompileCounterWithBackend("inductor")
        B, H, D = 1, 2, 64

        compiled = _compile_flex(counter)
        for q_len, kv_len in [(257, 513), (300, 600), (129, 257)]:
            bm = create_block_mask(_causal_mask, B=B, H=H, Q_LEN=q_len, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, q_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True,
                             tag=f"Q={q_len},KV={kv_len} both-uneven")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across both-unaligned metadata capacities, "
            f"got {counter.frame_count}"
        )

    def test_c6_backward_kv_variation(self, npu_device):
        """C6: backward smoke checks across exact dynamic-KV masks."""
        counter = CompileCounterWithBackend("inductor")
        B, H, Q, D = 1, 2, 256, 64

        compiled = _compile_flex(counter)
        for kv_len in [129, 512, 1024]:
            bm = create_block_mask(noop_mask, B=B, H=H, Q_LEN=Q, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"Q={Q},KV={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across backward KV capacities, "
            f"got {counter.frame_count}"
        )

    def test_c7_kv_len_reuse_with_backward(self, npu_device):
        """C7: exact masks for KV=512,1024,960,2048 reuse one graph."""
        counter = CompileCounterWithBackend("inductor")
        B, H, Q, D = 1, 2, 512, 64

        compiled = _compile_flex(counter)
        for kv_len in [512, 1024, 960, 2048]:
            bm = create_block_mask(noop_mask, B=B, H=H, Q_LEN=Q, KV_LEN=kv_len, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, kv_len, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"Q={Q},KV={kv_len}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across exact KV masks, got {counter.frame_count}"
        )
