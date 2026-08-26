"""Shared fixtures and helpers for score_mod dynamic-shape tests (dynamic-fix-new).

Mirrors dynamic-fix/test_flex_attention_dynamic_shape.py structure but:
  - Adds 11 score_mods (identity/times_two/squared/causal/inverse_causal/
    rel_bias/rel_causal/alibi_bias/trig/trig2/head_offset)
  - Dense SDPA reference with score_fn applied (matches compiled flex_attention
    score_mod semantics) so reference stays stable on NPU.
  - atol=5e-3, rtol=5e-3 (SD series standard).
  - autouse _reset_dynamo_cache to isolate CompileCounterWithBackend across tests.

Reference: dense fp32 SDPA with score_fn(scores, q_idx, kv_idx) applied.
"""
import math

import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    noop_mask,
)
from torch._dynamo.testing import CompileCounterWithBackend


# NPU device validation monkey-patch: bypass flex_attention's CUDA/CPU-only
# check so eager reference calls (used by head_offset fallback) work on NPU.
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass


# ============================================================================
# Tolerances (SD series standard)
# ============================================================================
SD_ATOL = 5e-3
SD_RTOL = 5e-3


# ============================================================================
# Defaults
# ============================================================================
DEFAULT_H_Q = 16
DEFAULT_H_KV = 16
DEFAULT_D = 64


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture(scope="session")
def npu_device():
    assert torch.npu.is_available(), "NPU is required for score_mod dynamic tests"
    return "npu"


@pytest.fixture(autouse=True)
def _reset_dynamo_cache():
    """Isolate Dynamo global cache across tests so CompileCounterWithBackend
    is reliably triggered per test (prevents cross-test cache hits that would
    make frame_count == 0)."""
    import torch._dynamo
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ============================================================================
# score_mod definitions (signature: (score, b, h, q_idx, kv_idx) -> score)
# 11 score_mods mirroring test_score_mod_only_npu.py
# ============================================================================
def identity_score_mod(score, b, h, q_idx, kv_idx):
    return score


def times_two_score_mod(score, b, h, q_idx, kv_idx):
    return score * 2


def squared_score_mod(score, b, h, q_idx, kv_idx):
    return score * score


def causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def inverse_causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx <= kv_idx, score, float("-inf"))


def rel_bias_score_mod(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


def rel_causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score + (q_idx - kv_idx), float("-inf"))


def alibi_bias_score_mod(score, b, h, q_idx, kv_idx):
    """ALiBi bias: scale = exp2(-(h+1)*8/num_heads), num_heads=8 (matches static test)."""
    num_heads = 8
    scale = torch.exp2(-((h + 1) * 8.0 / num_heads))
    return score + (kv_idx - q_idx) * scale


def trig_score_mod(score, b, h, q_idx, kv_idx):
    return torch.sin(torch.cos(score)) + torch.tan(b)


def trig2_score_mod(score, b, h, q_idx, kv_idx):
    cos_score = torch.cos(score)
    sin_score = torch.sin(score)
    return cos_score * sin_score + torch.tan(b)


# head_offset needs a captured buffer; provide a factory mirroring static test
_HEAD_OFFSET_CACHE = {}


def _get_head_offset(num_heads, device, dtype):
    key = (num_heads, str(device), str(dtype))
    if key not in _HEAD_OFFSET_CACHE:
        torch.manual_seed(0)
        _HEAD_OFFSET_CACHE[key] = torch.rand(num_heads, device=device, dtype=dtype)
    return _HEAD_OFFSET_CACHE[key]


def make_head_offset_score_mod(num_heads, device, dtype):
    """captured buffer: score * head_offset[h]. Static buffer to avoid
    _NPU_EXPLICIT_SCORE_MOD limitation on forward-generated intermediates."""
    head_offset = _get_head_offset(num_heads, device, dtype)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score * head_offset[h]

    return score_mod


# Registry (head_offset handled separately because it needs per-shape buffer)
SCORE_MOD_REGISTRY = {
    "identity": identity_score_mod,
    "times_two": times_two_score_mod,
    "squared": squared_score_mod,
    "causal": causal_score_mod,
    "inverse_causal": inverse_causal_score_mod,
    "rel_bias": rel_bias_score_mod,
    "rel_causal": rel_causal_score_mod,
    "alibi_bias": alibi_bias_score_mod,
    "trig": trig_score_mod,
    "trig2": trig2_score_mod,
    # head_offset is parameterized separately (needs factory)
}

ALL_SCORE_MOD_NAMES = list(SCORE_MOD_REGISTRY.keys()) + ["head_offset"]


def resolve_score_mod(name, *, num_heads=None, device=None, dtype=None):
    """Resolve score_mod by name; head_offset requires num_heads/device/dtype."""
    if name == "head_offset":
        assert num_heads is not None and device is not None and dtype is not None
        return make_head_offset_score_mod(num_heads, device, dtype)
    return SCORE_MOD_REGISTRY[name]


# ============================================================================
# Dense SDPA reference with score_fn applied
# ============================================================================
def _dense_score_fn_identity(scores, q_idx, kv_idx):
    return scores


def _dense_score_fn_times_two(scores, q_idx, kv_idx):
    return scores * 2


def _dense_score_fn_squared(scores, q_idx, kv_idx):
    return scores * scores


def _dense_score_fn_causal(scores, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, scores, float("-inf"))


def _dense_score_fn_inverse_causal(scores, q_idx, kv_idx):
    return torch.where(q_idx <= kv_idx, scores, float("-inf"))


def _dense_score_fn_rel_bias(scores, q_idx, kv_idx):
    return scores + (q_idx - kv_idx)


def _dense_score_fn_rel_causal(scores, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, scores + (q_idx - kv_idx), float("-inf"))


def _dense_score_fn_alibi_bias(scores, q_idx, kv_idx, h=None):
    """ALiBi applied per-head; needs h dimension so reference loops over heads."""
    # Note: this version returns scores for a single head; caller must supply h.
    # h is python int in dense reference context; use math.exp2 (not torch.exp2)
    # because torch.exp2(float) raises TypeError in newer PyTorch.
    num_heads = 8
    scale = math.exp2(-((h + 1) * 8.0 / num_heads))
    return scores + (kv_idx - q_idx) * scale


def _dense_score_fn_trig(scores, q_idx, kv_idx, b=None):
    return torch.sin(torch.cos(scores)) + torch.tan(b)


def _dense_score_fn_trig2(scores, q_idx, kv_idx, b=None):
    cos_s = torch.cos(scores)
    sin_s = torch.sin(scores)
    return cos_s * sin_s + torch.tan(b)


DENSE_SCORE_FN_REGISTRY = {
    "identity": _dense_score_fn_identity,
    "times_two": _dense_score_fn_times_two,
    "squared": _dense_score_fn_squared,
    "causal": _dense_score_fn_causal,
    "inverse_causal": _dense_score_fn_inverse_causal,
    "rel_bias": _dense_score_fn_rel_bias,
    "rel_causal": _dense_score_fn_rel_causal,
    # alibi_bias / trig / trig2 / head_offset handled specially (per-b/per-h)
}


def dense_reference(q, k, v, *, score_mod_name, head_offset_buffer=None):
    """Dense SDPA reference that mirrors the score_mod semantics.

    For score_mods that depend on b or h (alibi_bias uses h, trig/trig2 use b,
    head_offset uses h with captured buffer), loop over the corresponding axis
    so the reference matches compiled flex_attention exactly.

    Args:
        q, k, v: [B, H, Q, D] / [B, H, KV, D] / [B, H, KV, D] (fp32 promoted)
        score_mod_name: name from ALL_SCORE_MOD_NAMES
        head_offset_buffer: required for "head_offset"; shape [H]
    """
    B, H, Q, D = q.shape
    _, _, KV, _ = k.shape
    q_idx = torch.arange(Q, device=q.device, dtype=torch.float32)[:, None]  # [Q,1]
    kv_idx = torch.arange(KV, device=k.device, dtype=torch.float32)[None, :]  # [1,KV]

    # Compute scores per (b, h) for score_mods that need them; otherwise vectorize.
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    scale = 1.0 / math.sqrt(D)

    out = torch.empty(B, H, Q, D, dtype=torch.float32, device=q.device)

    for b in range(B):
        for h in range(H):
            scores = torch.matmul(q_f[b, h], k_f[b, h].transpose(-2, -1)) * scale  # [Q, KV]
            if score_mod_name == "identity":
                pass
            elif score_mod_name == "times_two":
                scores = scores * 2
            elif score_mod_name == "squared":
                scores = scores * scores
            elif score_mod_name == "causal":
                scores = torch.where(q_idx >= kv_idx, scores, float("-inf"))
            elif score_mod_name == "inverse_causal":
                scores = torch.where(q_idx <= kv_idx, scores, float("-inf"))
            elif score_mod_name == "rel_bias":
                scores = scores + (q_idx - kv_idx)
            elif score_mod_name == "rel_causal":
                scores = torch.where(q_idx >= kv_idx, scores + (q_idx - kv_idx), float("-inf"))
            elif score_mod_name == "alibi_bias":
                num_heads = 8
                # h is python int here (dense reference loops over b, h);
                # use math.exp2 (not torch.exp2) because torch.exp2(float)
                # raises TypeError in newer PyTorch.
                scale_h = math.exp2(-((h + 1) * 8.0 / num_heads))
                scores = scores + (kv_idx - q_idx) * scale_h
            elif score_mod_name == "trig":
                scores = torch.sin(torch.cos(scores)) + torch.tan(torch.tensor(float(b)))
            elif score_mod_name == "trig2":
                scores = torch.cos(scores) * torch.sin(scores) + torch.tan(torch.tensor(float(b)))
            elif score_mod_name == "head_offset":
                scores = scores * head_offset_buffer[h]
            else:
                raise ValueError(f"Unknown score_mod: {score_mod_name}")
            attn = torch.softmax(scores, dim=-1)
            # PyTorch SDPA / flex_attention use safe-softmax semantics: rows
            # where every score is -inf (e.g. inverse_causal when Q > KV) yield
            # 0 instead of NaN. Plain torch.softmax produces NaN on such rows;
            # explicitly zero them out so the dense reference matches kernel
            # behavior in degenerate shapes (C: Q=512 > KV=256 + inverse_causal).
            attn = torch.nan_to_num(attn, nan=0.0)
            out[b, h] = torch.matmul(attn, v_f[b, h])
    return out.to(q.dtype)


# ============================================================================
# Compile helper (mirrors dynamic-fix/_compile_flex_with_mask)
# ============================================================================
def compile_flex_with(counter, *, score_mod, block_mask, enable_gqa=False):
    """Bind score_mod + block_mask to a fresh partial; share counter so Dynamo
    reuses one cached graph across shapes (frame_count == 1)."""
    import functools
    attention = functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )
    compiled = torch.compile(attention, backend=counter, dynamic=True)

    def _call(q, k, v):
        return compiled(q, k, v)

    return _call


def compile_flex(counter, *, score_mod, enable_gqa=False):
    """Compile flex_attention with score_mod bound but block_mask passed at
    call time (mirrors px_cases ``_compile_flex`` pattern). Single compiled
    function is reused across all shapes; frame_count == 1.
    """
    def attention(q, k, v, block_mask):
        return flex_attention(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=enable_gqa,
        )

    return torch.compile(attention, backend=counter, dynamic=True)


# ============================================================================
# Assert helpers
# ============================================================================
def assert_close(actual, expected, tag, *, atol=SD_ATOL, rtol=SD_RTOL):
    """Compare tensors while preserving PyTorch's mismatch diagnostics.

    Passing ``msg`` directly to ``torch.testing.assert_close`` replaces its
    default diagnostic in some supported PyTorch versions.  Catch the failure
    instead so the test tag is prepended to details such as mismatched element
    count and greatest absolute/relative differences.
    """
    try:
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), atol=atol, rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(f"{tag}: forward output mismatch\n{exc}") from exc


def assert_close_with_details(actual, expected, tag, *, atol=SD_ATOL, rtol=SD_RTOL):
    """Generic version used for gradients and other non-forward comparisons."""
    try:
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), atol=atol, rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(f"{tag}\n{exc}") from exc


def assert_frame_count(counter, tag):
    assert counter.frame_count == 1, (
        f"{tag}: expected 1 compile (graph reuse), got {counter.frame_count}"
    )


def assert_grad_finite(tensor, name, tag):
    assert tensor.grad is not None, f"{tag}: missing {name} gradient"
    assert tuple(tensor.grad.shape) == tuple(tensor.shape), (
        f"{tag}: {name} grad shape {tensor.grad.shape} != {tensor.shape}"
    )
    assert torch.isfinite(tensor.grad).all().item(), f"{tag}: non-finite {name} grad"
