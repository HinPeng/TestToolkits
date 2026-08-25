"""Category B: dynamic Q_LEN (Q varies, B=2, KV=Q) × 11 score_mods.

Mirrors dynamic-fix/test_flex_attention_b_batch_envelope.py but adds 11
score_mods. Q=KV keeps square attention shape (matches static test's
seq_length pattern); Q spans mid-size LLM lengths to avoid BlockMask
capacity bucket edges.

Pattern:
- Each Q_LEN gets an exact, broadcastable ``B=1, H=1`` BlockMask captured
  by ``functools.partial`` together with ``score_mod``.
- All compiled partials share one ``CompileCounterWithBackend`` backend;
  Dynamo cache reuse => ``frame_count == 1``.
- Dense SDPA reference; atol=5e-3, rtol=5e-3.
- Forward only.

Run:
    pytest test_score_mod_b_q_len_dynamic.py -v
"""
import pytest
import torch

from torch.nn.attention.flex_attention import create_block_mask, noop_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_score_mod_dynamic_common import (
    npu_device,            # noqa: F401
    DEFAULT_H_Q,
    DEFAULT_D,
    ALL_SCORE_MOD_NAMES,
    resolve_score_mod,
    dense_reference,
    compile_flex_with,
    assert_close,
    assert_frame_count,
    _get_head_offset,
)


# Q_LEN spans 256..2048 (aligned multiples of 128) to avoid capacity bucket edges.
Q_LEN_SHAPES = [256, 512, 1024, 2048]
# Fixed B (exclude 1 to avoid Dynamo 0/1 specialization).
B_FIXED = 2


@pytest.mark.parametrize("score_mod_name", ALL_SCORE_MOD_NAMES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_b_q_len_dynamic(npu_device, dtype, score_mod_name):
    """B: Q_LEN in [256,512,1024,2048] dynamic, B=2, KV=Q, score_mod × dtype.

    Validates one reused graph across Q_LEN variations for each score_mod.
    """
    H, D = DEFAULT_H_Q, DEFAULT_D
    counter = CompileCounterWithBackend("inductor")

    head_offset_buffer = None
    if score_mod_name == "head_offset":
        head_offset_buffer = _get_head_offset(H, npu_device, dtype)
    score_mod = resolve_score_mod(
        score_mod_name, num_heads=H, device=npu_device, dtype=dtype
    )

    for Q in Q_LEN_SHAPES:
        bm = create_block_mask(
            noop_mask, B=1, H=1, Q_LEN=Q, KV_LEN=Q, device=npu_device,
        )
        compiled = compile_flex_with(counter, score_mod=score_mod, block_mask=bm)
        q = torch.randn(B_FIXED, H, Q, D, device=npu_device, dtype=dtype)
        k = torch.randn(B_FIXED, H, Q, D, device=npu_device, dtype=dtype)
        v = torch.randn(B_FIXED, H, Q, D, device=npu_device, dtype=dtype)
        actual = compiled(q, k, v)
        expected = dense_reference(
            q, k, v, score_mod_name=score_mod_name,
            head_offset_buffer=head_offset_buffer,
        )
        assert_close(actual, expected, f"B {score_mod_name} Q={Q} {dtype}")

    assert_frame_count(counter, f"B {score_mod_name}")
