"""Category D: combined dynamic B x Q x KV with shape-specific BlockMasks.

Each runtime Q/KV shape gets an exact, broadcastable ``B=1, H=1`` BlockMask
passed to one compiled function. Initial dynamic dimensions differ from static
``H=2``, and metadata capacities stay above one, so compile counts measure
torch_npu graph reuse rather than PyTorch 2.7 first-value specialization.

Run:
    pytest test_flex_attention_d_capacity_envelope.py -v
"""
import random

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
    _rel_bias_score_mod,
)


def _compile_flex(counter, score_mod=None):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestCombinedCapacityEnvelope:
    """D1-D4: exact combined-shape metadata capacities reuse one graph."""

    def test_d1_combined_shapes_single_graph(self, npu_device):
        """D1: exact masks while B, Q and KV change together."""
        counter = CompileCounterWithBackend("inductor")
        H, D = 2, 64

        compiled = _compile_flex(counter)
        for B, Q, KV in [(3, 512, 512), (4, 1024, 512), (5, 512, 2048), (8, 512, 512)]:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"B={B},Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across combined metadata capacities, "
            f"got {counter.frame_count}"
        )

    def test_d2_random_shapes(self, npu_device):
        """D2: 15 bounded random exact masks reuse one graph."""
        rng = random.Random(2026)
        H, D = 2, 64

        q_pool = [129, 256, 257, 512, 1024]
        kv_pool = [129, 256, 512, 513, 1024]
        shapes = [(rng.randint(3, 8), rng.choice(q_pool), rng.choice(kv_pool)) for _ in range(15)]

        counter = CompileCounterWithBackend("inductor")

        compiled = _compile_flex(counter)
        for B, Q, KV in shapes:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=False, tag=f"B={B},Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile for 15 random exact masks, "
            f"got {counter.frame_count}"
        )

    def test_d3_score_mod_path_dynamic(self, npu_device):
        """D3: score_mod path with exact dynamic-shape masks."""
        counter = CompileCounterWithBackend("inductor")
        H, D = 2, 64

        def rel_bias_ref(scores, m_idx, n_idx):
            return scores + (m_idx - n_idx)

        compiled = _compile_flex(counter, _rel_bias_score_mod)
        for B, Q, KV in [(3, 512, 512), (4, 512, 1024), (5, 1024, 512)]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, score_fn=rel_bias_ref,
                             tag=f"score_mod B={B},Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile on the dynamic score_mod path, "
            f"got {counter.frame_count}"
        )

    def test_d4_llm_infer_style_sequence(self, npu_device):
        """D4: serving-style sequence with one exact mask per step."""
        counter = CompileCounterWithBackend("inductor")
        H, D = 2, 64

        compiled = _compile_flex(counter)
        for B, Q, KV in [(3, 1024, 1024), (4, 1536, 2048), (5, 2048, 4096), (3, 1024, 1024)]:
            bm = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=Q, KV_LEN=KV, device=npu_device)
            q = torch.randn(B, H, Q, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, KV, D, device=npu_device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            _check_one_shape(compiled, q, k, v, bm, causal_ref=True, tag=f"B={B},Q={Q},KV={KV}")

        assert counter.frame_count == 1, (
            f"Expected 1 compile across the serving sequence, "
            f"got {counter.frame_count}"
        )
