"""Category Q: base score_mod/mask_mod scenarios with shape-specific BlockMasks.

Each runtime ``(B, S)`` shape gets an exact, broadcastable ``B=1, H=1``
BlockMask captured by ``functools.partial`` (together with ``score_mod``).
All compiled partials share one backend and must reuse one graph across
different B/S metadata capacities, matching community M01.

The reference is the dense SDPA math (``_dense_reference``) per shape,
matching S: avoids comparing flex_attention against itself (eager on NPU
takes an unoptimized path after bypassing ``_validate_device``, which
made identity/causal combos exceed tolerance).

H/head_dim are static dims (fixed at H=8, D=64); only B and S vary.
S >= 512 avoids the known flex_decode path issue at S <= 256
(create_flex_decoding_kernel calls torch.cuda.get_device_capability).

Run:
    pytest test_flex_attention_q_base_scenario_envelope.py -v
"""
import functools

import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _BASE_SCORE_MODS,
    _BASE_MASK_MODS,
    _dense_reference,
)

_H, _D = 8, 64
_RUNTIME_SHAPES = [(2, 512), (2, 1024), (4, 2048)]


def _wrap_score_mod(score_mod):
    """Adapt a (score, b, h, m, n) score_mod to ``_dense_reference``'s
    ``(scores, q_idx, kv_idx) -> scores`` ``score_fn`` signature.

    All ``_BASE_SCORE_MODS`` (identity/causal/rel_bias) ignore b/h, so
    passing 0/0 is valid. ``q_idx`` shape ``[Q, 1]`` and ``kv_idx`` shape
    ``[1, KV]`` broadcast against ``scores`` shape ``[B, H, Q, KV]``.
    """
    def score_fn(scores, q_idx, kv_idx):
        return score_mod(scores, 0, 0, q_idx, kv_idx)
    return score_fn


def _dense_ref_for(q, k, v, score_mod, mask_mod):
    """Dense fp32 SDPA reference for the Q combos.

    ``_BASE_MASK_MODS`` (noop/causal) are b/h-independent, so
    ``_dense_reference``'s ``b=0/h=0`` sampling is valid for all B/H.
    """
    return _dense_reference(
        q, k, v,
        causal=False,
        score_fn=_wrap_score_mod(score_mod),
        mask_fn=mask_mod,
    )


def _compile_flex_with_mask(counter, block_mask, score_mod=None):
    attention = functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
    )
    compiled = torch.compile(attention, backend=counter, dynamic=True)

    def call(q, k, v, _unused_block_mask):
        return compiled(q, k, v)

    return call


class TestBaseScenarioEnvelope:
    """Q: base score_mod x mask_mod scenarios, exact masks reuse one graph."""

    @pytest.mark.parametrize("score_mod_name", list(_BASE_SCORE_MODS.keys()))
    @pytest.mark.parametrize("mask_mod_name", list(_BASE_MASK_MODS.keys()))
    def test_base_mods_multi_shape_single_graph(self, npu_device, score_mod_name,
                                                mask_mod_name):
        """Exact BlockMask per (B, S) shape, captured via partial; all
        compiled partials share one backend -> 1 compile.
        """
        score_mod = _BASE_SCORE_MODS[score_mod_name]
        mask_mod = _BASE_MASK_MODS[mask_mod_name]

        counter = CompileCounterWithBackend("inductor")

        for B, S in _RUNTIME_SHAPES:
            bm = create_block_mask(mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
            compiled = _compile_flex_with_mask(counter, bm, score_mod)
            q = torch.randn(B, _H, S, _D, device=npu_device, dtype=torch.float32)
            k = torch.randn(B, _H, S, _D, device=npu_device, dtype=torch.float32)
            v = torch.randn_like(k)

            # Dense fp32 SDPA reference (no flex_attention self-compare).
            ref = _dense_ref_for(q, k, v, score_mod, mask_mod)

            out = compiled(q, k, v, bm)
            torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3,
                                       msg=f"{score_mod_name} x {mask_mod_name} @ B={B},S={S}")

        assert counter.frame_count == 1, (
            f"{score_mod_name} x {mask_mod_name}: expected 1 compile across exact masks, "
            f"got {counter.frame_count}"
        )
