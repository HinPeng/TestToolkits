"""Shared fixtures and bounded correctness checks for dynamic-shape tests."""

import math

import pytest
import torch

from torch.nn.attention.flex_attention import noop_mask


# NPU device validation monkey-patch: bypass flex_attention's CUDA/CPU-only
# check so eager reference calls (used by L/M/Q envelope tests) work on NPU.
# torch_npu doesn't patch this upstream across all versions.
try:
    from torch.nn.attention import flex_attention as _fa_module
    _fa_module._validate_device = lambda q, k, v: None
except Exception:
    pass


@pytest.fixture(scope="session")
def npu_device():
    assert torch.npu.is_available(), "NPU is required for the Flex Attention tests"
    return "npu"


def _causal_mask(batch, head, token_q, token_kv):
    return token_q >= token_kv


def _rel_bias_score_mod(score, batch, head, token_q, token_kv):
    return score + (token_q - token_kv)


def _base_identity_score_mod(score, b, h, m, n):
    return score


def _base_causal_score_mod(score, b, h, m, n):
    return torch.where(m >= n, score, float("-inf"))


def _base_rel_bias_score_mod(score, b, h, m, n):
    return score + (m - n)


def _base_causal_mask_mod(b, h, m, n):
    return m >= n


_BASE_SCORE_MODS = {
    "identity": _base_identity_score_mod,
    "causal": _base_causal_score_mod,
    "rel_bias": _base_rel_bias_score_mod,
}

_BASE_MASK_MODS = {
    "noop": noop_mask,
    "causal": _base_causal_mask_mod,
}

_BASE_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _times_two_score(score, b, h, m, n):
    return score * 2


def _squared_score(score, b, h, m, n):
    return score * score


def _inverse_causal_score(score, b, h, m, n):
    return torch.where(m <= n, score, float("-inf"))


def _alibi_bias_score(score, b, h, m, n):
    scale = torch.exp2(-((h + 1) * 8.0 / 16))
    return score + (n - m) * scale


def _rel_causal_score(score, b, h, m, n):
    return torch.where(m >= n, score + (m - n), float("-inf"))


def _trig_score(score, b, h, m, n):
    return torch.sin(torch.cos(score)) + torch.tan(b)


def _silu_score(score, b, h, m, n):
    return torch.nn.functional.silu(score)


def _skip_even_keys_score(score, b, h, m, n):
    return torch.where(n % 2 == 0, score, float("-inf"))


def _head_scale_score(score, b, h, m, n):
    return score * (1.0 + h * 0.1)


def _windowed_mask(b, h, m, n):
    return (m - n).abs() <= 64


def _block_diagonal_mask(b, h, m, n):
    return (m // 128) == (n // 128)


def _head_dependent_mask(b, h, m, n):
    return (h % 2 == 0) | (m >= n)


def _inverse_causal_mask(b, h, m, n):
    return m <= n


def _prefix_mask(b, h, m, n):
    return n < 128


def _block_diagonal_mask_64(b, h, m, n):
    return (m // 64) == (n // 64)


def _causal_with_prefix_mask(b, h, m, n):
    return (m >= n) | (n < 32)


_COMBO_SCORE_MODS = {
    "identity": _base_identity_score_mod,
    "times_two": _times_two_score,
    "squared": _squared_score,
    "inverse_causal": _inverse_causal_score,
    "alibi_bias": _alibi_bias_score,
    "rel_causal": _rel_causal_score,
    "trig": _trig_score,
    "silu": _silu_score,
    "skip_even_keys": _skip_even_keys_score,
    "head_scale": _head_scale_score,
}

_COMBO_MASK_MODS = {
    "noop": noop_mask,
    "causal": _base_causal_mask_mod,
    "inverse_causal": _inverse_causal_mask,
    "windowed": _windowed_mask,
    "prefix": _prefix_mask,
    "block_diagonal": _block_diagonal_mask_64,
    "head_dependent": _head_dependent_mask,
    "causal_with_prefix": _causal_with_prefix_mask,
}

_EXPECTED_FAILURE_COMBOS = {("inverse_causal", "causal")}

_VALID_COMBOS = [
    (score_name, mask_name)
    for score_name in _COMBO_SCORE_MODS
    for mask_name in _COMBO_MASK_MODS
    if (score_name, mask_name) not in _EXPECTED_FAILURE_COMBOS
]


def _remainder_mask_mod(b, h, m, n):
    """Uses % operator - triggers aten.remainder.Scalar."""
    return (h % 2 == 0) | (m >= n)


def _remainder_mod3_mask(b, h, m, n):
    """Uses % 3."""
    return (h % 3 == 0) | (m >= n)


def _remainder_boolean_mask(b, h, m, n):
    """Uses % with an explicit boolean comparison."""
    return (h % 2 != 0) | (m >= n)


def _remainder_in_score_mod(score, b, h, m, n):
    """Uses % in score_mod - should work."""
    return torch.where(n % 2 == 0, score, float("-inf"))


def _bitwise_and_mask(b, h, m, n):
    """Uses & 1 instead of % 2 - should work."""
    return (h & 1 == 0) | (m >= n)


def _equiv_form_mask(b, h, m, n):
    """Equivalent to h % 2 == 0 using // and *."""
    return (h - (h // 2) * 2 == 0) | (m >= n)


def _floordiv_mask(b, h, m, n):
    """Uses // operator."""
    return (m // 64) == (n // 64)


FWD_ATOL = 2e-2
FWD_RTOL = 2e-2
GRAD_ATOL = 8e-2
GRAD_RTOL = 8e-2


def assert_close_with_details(actual, expected, *, atol, rtol, msg=None, **kwargs):
    """Keep PyTorch mismatch statistics while adding test-case context.

    Passing ``msg=`` directly to ``torch.testing.assert_close`` replaces its
    useful default text in some supported PyTorch versions. Calling it without
    ``msg`` first preserves details such as mismatched element count, greatest
    absolute difference, and greatest relative difference; the case label is
    then prepended only when the comparison fails.
    """

    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, **kwargs)
    except AssertionError as exc:
        if not msg:
            raise
        raise AssertionError(f"{msg}\n{exc}") from exc


def _dense_reference(q, k, v, *, causal, score_fn, mask_fn=None):
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    q_idx = torch.arange(q_len, device=q.device, dtype=torch.float32)[:, None]
    kv_idx = torch.arange(kv_len, device=k.device, dtype=torch.float32)[None, :]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
    if score_fn is not None:
        scores = score_fn(scores, q_idx, kv_idx)
    if mask_fn is not None:
        b_idx = torch.zeros(1, dtype=torch.long, device=q.device)
        h_idx = torch.zeros(1, dtype=torch.long, device=q.device)
        keep = mask_fn(b_idx, h_idx, q_idx.long(), kv_idx.long())
        scores = scores.masked_fill(~keep, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def _check_one_shape(compiled, q, k, v, block_mask, *, causal_ref, tag, score_fn=None):
    actual = compiled(q, k, v, block_mask)
    torch.npu.synchronize()
    assert tuple(actual.shape) == tuple(q.shape), f"{tag}: output shape {actual.shape} != {q.shape}"
    assert torch.isfinite(actual).all().item(), f"{tag}: non-finite forward output"

    if q.shape[-2] * k.shape[-2] <= 1024 * 1024:
        expected = _dense_reference(q, k, v, causal=causal_ref, score_fn=score_fn)
        assert_close_with_details(
            actual,
            expected,
            atol=8e-2,
            rtol=8e-2,
            msg=tag,
        )

    actual.float().sum().backward()
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        assert tensor.grad is not None, f"{tag}: missing {name} gradient"
        assert tuple(tensor.grad.shape) == tuple(tensor.shape), (
            f"{tag}: {name} gradient shape {tensor.grad.shape} != {tensor.shape}"
        )
        assert torch.isfinite(tensor.grad).all().item(), f"{tag}: non-finite {name} gradient"
