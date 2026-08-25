"""SD1-SD4: score_mod dimension x dynamic shape.

Validates that head/batch-dependent score_mods still work when the compiled
graph is reused across multiple runtime shapes (B / Q / KV / S dynamic) under
``torch.compile(dynamic=True)``.

Pattern (mirrors dynamic-fix/A-D):
- Each runtime shape gets its own exact-sized ``B=1, H=1`` broadcastable
  BlockMask passed as an input to one compiled function.
- Each test compiles once outside the shape loop; Dynamo graph reuse gives
  ``frame_count == 1`` (asserted at the end of each test).
- Dense SDPA (fp32 math) is the ground truth.
- Tolerance: atol=5e-3, rtol=5e-3.

Run:
    pytest test_sd_score_mod_dynamic.py -v
"""
import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import assert_close_with_details
from test_flex_attention_score_mode_common import (
    npu_device,            # noqa: F401  (pytest fixture re-export)
    DEFAULT_B,
    DEFAULT_H,
    DEFAULT_D,
    SD_ATOL,
    SD_RTOL,
    alibi_bias_score_mod,
    rel_bias_score_mod,
    composed_rel_causal_score_mod,
    make_captured_buffer_score_mod,
    windowed_mask_mod_factory,
    causal_mask_mod,
    dense_reference,
)


def _compile_with(counter, *, score_mod=None):
    """Compile once while keeping shape-specific BlockMasks as inputs."""
    def attention(q, k, v, block_mask):
        return flex_attention(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )

    return torch.compile(attention, backend=counter, dynamic=True)


def _assert_close(actual, expected, tag):
    assert_close_with_details(
        actual,
        expected,
        atol=SD_ATOL,
        rtol=SD_RTOL,
        msg=f"{tag}: forward output mismatch",
    )


def _assert_frame_count(counter, tag):
    assert counter.frame_count == 1, (
        f"{tag}: expected 1 compile (graph reuse), got {counter.frame_count}"
    )


# ============================================================================
# SD1: alibi_bias (head-dependent) x Q_LEN dynamic
# ============================================================================
def _alibi_dense_score_fn(scores, q_idx, kv_idx):
    """Dense-ref adapter for alibi: scores is [B,H,Q,KV], apply per-head scale."""
    H = scores.shape[1]
    h_idx = torch.arange(H, device=scores.device, dtype=torch.float32)
    scale = torch.exp2(-((h_idx + 1) * 8.0 / 16))  # [H]
    return scores + (kv_idx - q_idx) * scale.view(1, -1, 1, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sd1_alibi_q_dynamic(npu_device, dtype):
    """SD1: alibi bias + causal mask + Q varies 256->384->512. Frame count 1."""
    B, H, D = DEFAULT_B, DEFAULT_H, DEFAULT_D
    Q_SHAPES = [256, 384, 512]
    counter = CompileCounterWithBackend("inductor")
    compiled = _compile_with(counter, score_mod=alibi_bias_score_mod)
    for Q in Q_SHAPES:
        bm = create_block_mask(causal_mask_mod, B=1, H=1, Q_LEN=Q, KV_LEN=Q,
                               device=npu_device)
        q = torch.randn(B, H, Q, D, device=npu_device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        actual = compiled(q, k, v, bm)
        expected = dense_reference(q, k, v, causal=True, score_fn=_alibi_dense_score_fn)
        _assert_close(actual, expected, f"SD1 Q={Q} {dtype}")
    _assert_frame_count(counter, "SD1")


# ============================================================================
# SD2: rel_bias + windowed mask x KV_LEN dynamic
# ============================================================================
def _rel_bias_dense_score_fn(scores, q_idx, kv_idx):
    return scores + (q_idx - kv_idx)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sd2_rel_bias_windowed_kv_dynamic(npu_device, dtype):
    """SD2: rel_bias + windowed(100) + KV varies 256->384->512. Frame count 1."""
    B, H, D = DEFAULT_B, DEFAULT_H, DEFAULT_D
    KV_SHAPES = [256, 384, 512]
    counter = CompileCounterWithBackend("inductor")
    windowed = windowed_mask_mod_factory(100)
    compiled = _compile_with(counter, score_mod=rel_bias_score_mod)
    for KV in KV_SHAPES:
        bm = create_block_mask(windowed, B=1, H=1, Q_LEN=KV, KV_LEN=KV,
                               device=npu_device)
        q = torch.randn(B, H, KV, D, device=npu_device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        actual = compiled(q, k, v, bm)
        expected = dense_reference(q, k, v, causal=False,
                                  score_fn=_rel_bias_dense_score_fn, mask_fn=windowed)
        _assert_close(actual, expected, f"SD2 KV={KV} {dtype}")
    _assert_frame_count(counter, "SD2")


# ============================================================================
# SD3: composed (rel_bias + causal) + causal mask x (B, Q) joint dynamic
# ============================================================================
def _composed_dense_score_fn(scores, q_idx, kv_idx):
    """rel_bias then causal mask (matches composed_rel_causal_score_mod)."""
    return torch.where(q_idx >= kv_idx, scores + (q_idx - kv_idx), float("-inf"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sd3_composed_b_q_dynamic(npu_device, dtype):
    """SD3: composed score + causal mask + (B,Q) varies. Frame count 1."""
    H, D = DEFAULT_H, DEFAULT_D
    BQ_SHAPES = [(2, 256), (4, 384), (2, 512)]
    counter = CompileCounterWithBackend("inductor")
    compiled = _compile_with(counter, score_mod=composed_rel_causal_score_mod)
    for B, Q in BQ_SHAPES:
        bm = create_block_mask(causal_mask_mod, B=1, H=1, Q_LEN=Q, KV_LEN=Q,
                               device=npu_device)
        q = torch.randn(B, H, Q, D, device=npu_device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        actual = compiled(q, k, v, bm)
        # composed score already includes causal mask, so dense ref uses causal=False
        # and the score_fn applies both rel_bias and causal via torch.where.
        expected = dense_reference(q, k, v, causal=False,
                                  score_fn=_composed_dense_score_fn)
        _assert_close(actual, expected, f"SD3 B={B} Q={Q} {dtype}")
    _assert_frame_count(counter, "SD3")


# ============================================================================
# SD4: captured_buffer (head_offset) + noop mask x S dynamic
# ============================================================================
def _make_captured_dense_score_fn(head_offset):
    def _fn(scores, q_idx, kv_idx):
        return scores + head_offset.view(1, -1, 1, 1)
    return _fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sd4_captured_buffer_s_dynamic(npu_device, dtype):
    """SD4: score_mod captures external head_offset tensor + S varies. Frame 1."""
    from torch.nn.attention.flex_attention import noop_mask
    B, H, D = DEFAULT_B, DEFAULT_H, DEFAULT_D
    S_SHAPES = [256, 512, 1024]
    counter = CompileCounterWithBackend("inductor")
    head_offset = torch.rand(H, device=npu_device, dtype=dtype)
    score_mod = make_captured_buffer_score_mod(head_offset)
    compiled = _compile_with(counter, score_mod=score_mod)
    for S in S_SHAPES:
        bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=S, KV_LEN=S,
                               device=npu_device)
        q = torch.randn(B, H, S, D, device=npu_device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        actual = compiled(q, k, v, bm)
        dense_fn = _make_captured_dense_score_fn(head_offset)
        expected = dense_reference(q, k, v, causal=False, score_fn=dense_fn)
        _assert_close(actual, expected, f"SD4 S={S} {dtype}")
    _assert_frame_count(counter, "SD4")
