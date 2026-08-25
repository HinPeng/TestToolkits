"""Category O: head_dim variations with shape-specific BlockMasks.

head_dim stays parametrized (it configures the kernel tile and is not a
dynamic sequence dim); within each head_dim, each runtime ``S`` shape gets
an exact, broadcastable ``B=1, H=1`` BlockMask passed to one compiled
function. The graph must be reused across different S metadata capacities.

The eager reference is computed per shape with an EXACT-SIZED BlockMask.

Run:
    pytest test_flex_attention_o_head_dim_envelope.py -v
"""
import pytest
import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask
from torch._dynamo.testing import CompileCounterWithBackend

from test_flex_attention_dynamic_shape import (
    npu_device,          # noqa: F401  (pytest fixture re-export)
    _dense_reference,
)

_B, _H = 2, 8
_RUNTIME_S = [128, 192, 256]


def _compile_flex(counter):
    def attention(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    return torch.compile(attention, backend=counter, dynamic=True)


class TestHeadDimEnvelope:
    """O: head_dim alignment with exact dynamic-shape masks reusing one graph."""

    @pytest.mark.parametrize("head_dim", [16, 24, 32, 64, 80, 94, 96, 128])
    def test_head_dim(self, npu_device, head_dim):
        """Exact masks per head_dim over S = 128/192/256, fwd only."""
        dtype = torch.float32

        counter = CompileCounterWithBackend("inductor")
        compiled = _compile_flex(counter)

        for S in _RUNTIME_S:
            bm = create_block_mask(noop_mask, B=1, H=1, Q_LEN=S, KV_LEN=S,
                                   device=npu_device)
            q = torch.randn(_B, _H, S, head_dim, device=npu_device, dtype=dtype)
            k = torch.randn(_B, _H, S, head_dim, device=npu_device, dtype=dtype)
            v = torch.randn(_B, _H, S, head_dim, device=npu_device, dtype=dtype)

            # Eager reference with an exact-sized mask for this shape
            bm_ref = create_block_mask(noop_mask, B=_B, H=_H, Q_LEN=S, KV_LEN=S,
                                       device=npu_device)
            ref_out = flex_attention(q, k, v, block_mask=bm_ref)

            comp_out = compiled(q, k, v, bm)

            max_err = torch.max(torch.abs(ref_out - comp_out)).item()
            assert max_err <= 0.1, (
                f"head_dim={head_dim}, S={S}: max_err={max_err:.6f} > 0.1, "
                f"numerical corruption"
            )

        assert counter.frame_count == 1, (
            f"head_dim={head_dim}: expected 1 compile across exact masks, "
            f"got {counter.frame_count}"
        )
