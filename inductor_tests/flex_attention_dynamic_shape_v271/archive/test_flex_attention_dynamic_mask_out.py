import math
import importlib.util
import os
import unittest
from pathlib import Path


def _torch_npu_root() -> Path:
    """Locate either an installed torch_npu package or a source checkout."""

    configured = os.environ.get("TORCH_NPU_SOURCE_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.name == "torch_npu" and (root / "_inductor").is_dir():
            return root
        if (root / "torch_npu" / "_inductor").is_dir():
            return root / "torch_npu"
        raise RuntimeError(
            "TORCH_NPU_SOURCE_ROOT must point to a torch_npu package "
            "directory or its source-checkout parent"
        )

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate torch_npu; activate the torch_npu environment")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


TORCH_NPU_ROOT = _torch_npu_root()
LOWERING_PATH = TORCH_NPU_ROOT / "_inductor/kernel/flex_attention.py"
TEMPLATE_PATH = TORCH_NPU_ROOT / "_inductor/kernel/flexattention_template.py"
METADATA_PATH = TORCH_NPU_ROOT / "_inductor/kernel/flex_attention_metadata.py"


def _read(path):
    return path.read_text(encoding="utf-8")


try:
    import torch
    import torch_npu
    import torch_npu._inductor

    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    HAS_NPU = False


class TestFlexAttentionDynamicMaskOutSource(unittest.TestCase):
    def test_eager_compact_metadata_is_removed(self):
        lowering = _read(LOWERING_PATH)
        metadata = _read(METADATA_PATH)

        self.assertNotIn("COMPACT_SPARSE_MASK_TOTAL_BLOCKS", lowering)
        self.assertNotIn("_precompute_compact_sparse_mask_metadata", lowering)
        self.assertNotIn("_wrap_mask_mod_with_compact_sparse_mask_metadata", lowering)
        self.assertNotIn("_SPARSE_MASK_COMPACT_OPTION_KEYS", metadata)

    def test_compact_metadata_is_split_into_offsets_and_mapping(self):
        template = _read(TEMPLATE_PATH)

        self.assertIn("compute_compact_sparse_mask_offsets_kernel", template)
        self.assertIn("compute_compact_sparse_mask_mapping_kernel", template)
        self.assertIn("tl.atomic_add(TOTAL_BLOCKS", template)
        self.assertIn('{{size("KV_NUM_BLKS", 0)}}', template)
        self.assertIn('{{size("KV_NUM_BLKS", 2)}}', template)
        self.assertIn(
            '{{def_kernel("Q_OFFSETS", "TOTAL_BLOCKS", "KV_NUM_BLKS")}}',
            template,
        )
        self.assertIn(
            '{{def_kernel("FLAT_TO_ROW", "FLAT_TO_BLK", "Q_OFFSETS", '
            '"KV_NUM_BLKS")}}',
            template,
        )
        self.assertNotIn('manual_output_buffer="arg_TOTAL_BLOCKS"', template)

    def test_compact_kernels_use_symbolic_mapping_sizes(self):
        template = _read(TEMPLATE_PATH)

        self.assertEqual(template.count("tl.load(TOTAL_BLOCKS)"), 0)
        self.assertNotIn("TOTAL_FLAT_ENTRIES", template)
        self.assertNotIn("SPARSE_Z: tl.constexpr", template)
        self.assertNotIn("SPARSE_HQ: tl.constexpr", template)

    def test_forward_compact_kernel_uses_symbolic_mapping_sizes(self):
        template = _read(TEMPLATE_PATH)
        forward_compact = template.split(
            "compute_sparse_mask_kernel_compact =", 1
        )[1].split("compute_bwd_sparse_mask_kernel_compact =", 1)[0]
        forward_block = template.split(
            "compute_forward_block_mn_sparse_mask =", 1
        )[1].split("compute_forward_inner_sparse_mask_direct_index =", 1)[0]
        forward_kernel = template.split(
            "compute_flex_attention_sparse_mask_in_loop_no_load_balance =", 1
        )[1].split("compute_forward_block_mn_full =", 1)[0]

        self.assertIn('{{size("FLAT_TO_ROW", 0)}}', forward_compact)
        self.assertIn('{{size("Q", 2)}}', forward_compact)
        self.assertIn('{{size("K", 2)}}', forward_compact)
        self.assertNotIn("TOTAL_FLAT_ENTRIES", forward_compact)
        self.assertNotIn("SPARSE_MASK_HQ", forward_block)
        self.assertNotIn("NUM_SPARSE_Q_BLOCKS", forward_block + forward_kernel)

    def test_backward_compact_kernels_use_symbolic_mapping_sizes(self):
        template = _read(TEMPLATE_PATH)
        backward_compact = template.split(
            "compute_bwd_sparse_mask_kernel_compact =", 1
        )[1].split("compute_sparse_mask_block_pos_kernel =", 1)[0]
        block_pos = template.split(
            "compute_sparse_mask_block_pos_kernel =", 1
        )[1].split("compute_forward_block_mn_sparse_mask =", 1)[0]

        for source in (backward_compact, block_pos):
            self.assertIn('{{size("FLAT_TO_ROW", 0)}}', source)
            self.assertNotIn("TOTAL_FLAT_ENTRIES", source)
            self.assertNotIn("NUM_SPARSE_Q_BLOCKS", source)
        self.assertNotIn("SPARSE_MASK_HQ", template)
        self.assertNotIn("SPARSE_Z: tl.constexpr", template)
        self.assertNotIn("SPARSE_HQ: tl.constexpr", template)
        self.assertIn(
            'kv_sparse_idx * {{stride("SPARSE_MASK_BLOCK_POS", 3)}}',
            template,
        )

    def test_lowering_allocates_symbolic_actual_capacity(self):
        lowering = _read(LOWERING_PATH)

        self.assertIn("_build_runtime_compact_sparse_mask_offsets", lowering)
        self.assertIn("_build_runtime_compact_sparse_mask_mapping", lowering)
        self.assertIn("_bind_runtime_total_blocks_as_unbacked_size", lowering)
        self.assertIn("DynamicScalar(symbol, (), runtime_total_blocks)", lowering)
        self.assertIn("actual_blocks", lowering)
        self.assertNotIn("capacity_blocks", lowering)
        self.assertIn("sympy.prod(kv_num_blocks.get_size())", lowering)
        self.assertIn("lowerings[aten.fill_](total_blocks, 0)", lowering)
        self.assertIn("AssertScalar(", lowering)
        self.assertIn("pending_fresh_unbacked_symbols", lowering)

    def test_legacy_static_total_options_are_removed(self):
        lowering = _read(LOWERING_PATH)

        self.assertNotIn('"TOTAL_FLAT_ENTRIES"', lowering)

    def test_dynamic_backward_disables_static_tasklist_codegen(self):
        lowering = _read(LOWERING_PATH)

        self.assertIn("bwd_has_dynamic_shape", lowering)
        self.assertIn("or bwd_has_dynamic_shape", lowering)

    def test_symbolic_sequence_lengths_do_not_require_python_truth_values(self):
        lowering = _read(LOWERING_PATH)
        compact_lowering = "".join(lowering.split())

        self.assertNotIn("if seq_len_q % 128", lowering)
        self.assertIn(
            "statically_known_multiple_of(seq_len_q,128)", compact_lowering
        )
        self.assertIn(
            "statically_known_multiple_of(seq_len_kv,128)", compact_lowering
        )
        self.assertIn(
            "statically_known_lt(query.get_size()[-2],128)", compact_lowering
        )
        template = _read(TEMPLATE_PATH)
        forward_grid = template.split(
            "def flex_attention_in_loop_grid(", 1
        )[1].split("def compact_offsets_grid(", 1)[0]
        self.assertIn("*,cdiv,min", "".join(forward_grid.split()))

    def test_default_noop_block_mask_uses_mask_in_forward(self):
        lowering = _read(LOWERING_PATH)
        backward_lowering = lowering.split(
            "@register_lowering(torch.ops.higher_order.flex_attention_backward", 1
        )[1]

        self.assertIn("_is_default_noop_block_mask", lowering)
        self.assertIn("flexattention_mask_out = not is_default_noop_block_mask", lowering)
        self.assertIn("and not is_default_noop_block_mask", backward_lowering)

    def test_backward_dq_task_count_comes_from_runtime_q_shape(self):
        template = _read(TEMPLATE_PATH)

        dq_kernel = template.split(
            "flex_attention_backward_qmajor_dq_source =", 1
        )[1].split("flex_attention_backward_dkdv_only_source =", 1)[0]
        self.assertIn("DQ_NUM_Q_BLOCKS = tl.cdiv(Q_LEN, BLOCK_M2)", dq_kernel)
        self.assertIn("DQ_NUM_TASKS = DQ_NUM_Q_BLOCKS * ZQ * HQ", dq_kernel)

    def test_backward_block_position_covers_all_runtime_kv_blocks(self):
        lowering = _read(LOWERING_PATH)

        self.assertIn(
            "num_sparse_kv_blocks = (\n"
            "                seq_len_kv + SPARSE_KV_BLOCK_SIZE - 1\n"
            "            ) // SPARSE_KV_BLOCK_SIZE",
            lowering,
        )
        self.assertIn(
            "*kv_num_blocks.get_size(),\n"
            "                num_sparse_kv_blocks,",
            lowering,
        )


@unittest.skipUnless(HAS_NPU, "requires a built torch_npu package and NPU device")
class TestFlexAttentionDynamicMaskOutNPU(unittest.TestCase):
    @staticmethod
    def _causal_mask(_b, _h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @staticmethod
    def _dense_reference(q, k, v, *, causal):
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        scores = scores / math.sqrt(q.size(-1))
        if causal:
            q_idx = torch.arange(q.size(-2), device=q.device)[:, None]
            kv_idx = torch.arange(k.size(-2), device=k.device)[None, :]
            scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v.float()).to(q.dtype)

    def test_create_block_mask_has_no_compact_metadata_capture(self):
        from torch.nn.attention.flex_attention import create_block_mask

        block_mask = create_block_mask(
            self._causal_mask,
            B=2,
            H=2,
            Q_LEN=257,
            KV_LEN=193,
            device="npu",
        )
        self.assertFalse(
            hasattr(block_mask.mask_mod, "_npu_compact_sparse_mask_metadata")
        )
        cached = getattr(block_mask, "_npu_flex_attention_kernel_options", {})
        for key in (
            "COMPACT_SPARSE_MASK_TOTAL_BLOCKS",
            "SPARSE_MASK_MAX_NORMAL_BLOCKS",
            "SPARSE_MASK_HQ",
            "SPARSE_MASK_HEAD_SHARED",
        ):
            self.assertNotIn(key, cached)

    def test_forward_reuses_graph_for_dynamic_q_kv_and_sparse_z(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        counter = CompileCounterWithBackend("inductor")

        def fn(q, k, v, block_mask):
            return flex_attention(q, k, v, block_mask=block_mask)

        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
        for batch, q_len, kv_len in [(5, 257, 385), (6, 129, 257)]:
            q = torch.randn(batch, 2, q_len, 64, device="npu", dtype=torch.bfloat16)
            k = torch.randn(
                batch, 2, kv_len, 64, device="npu", dtype=torch.bfloat16
            )
            v = torch.randn_like(k)
            block_mask = create_block_mask(
                self._causal_mask,
                B=batch,
                H=2,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device="npu",
            )
            expected = self._dense_reference(q, k, v, causal=True)
            actual = compiled(q, k, v, block_mask)
            # NPU bfloat16 output can differ by two ULPs after accumulation.
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

        self.assertEqual(counter.frame_count, 1)

    def test_backward_reuses_graph_for_dynamic_q_kv_and_sparse_z(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        counter = CompileCounterWithBackend("inductor")

        def fn(q, k, v, block_mask):
            output = flex_attention(q, k, v, block_mask=block_mask)
            return output.float().square().mean()

        compiled = torch.compile(fn, backend=counter, dynamic=True, fullgraph=True)
        for batch, q_len, kv_len in [(5, 257, 385), (6, 129, 257)]:
            inputs = [
                torch.randn(
                    batch,
                    2,
                    length,
                    64,
                    device="npu",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                for length in (q_len, kv_len, kv_len)
            ]
            block_mask = create_block_mask(
                self._causal_mask,
                B=batch,
                H=2,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device="npu",
            )
            ref_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
            ref_output = self._dense_reference(
                *ref_inputs,
                causal=True,
            )
            ref_loss = ref_output.float().square().mean()
            ref_grads = torch.autograd.grad(ref_loss, ref_inputs)
            loss = compiled(*inputs, block_mask)
            grads = torch.autograd.grad(loss, inputs)
            torch.testing.assert_close(loss, ref_loss, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(grads, ref_grads, atol=8e-2, rtol=8e-2)

        self.assertEqual(counter.frame_count, 1)

    def test_full_mask_uses_runtime_compact_metadata(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def full_mask(_b, _h, q_idx, _kv_idx):
            return q_idx >= 0

        q = torch.randn(
            2,
            2,
            192,
            64,
            device="npu",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            2,
            2,
            256,
            64,
            device="npu",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        v = torch.randn_like(k, requires_grad=True)
        block_mask = create_block_mask(
            full_mask,
            B=2,
            H=2,
            Q_LEN=192,
            KV_LEN=256,
            device="npu",
        )
        self.assertGreater(block_mask.kv_num_blocks.numel(), 0)
        compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True)
        ref_inputs = [
            x.detach().clone().requires_grad_(True) for x in (q, k, v)
        ]
        expected = self._dense_reference(*ref_inputs, causal=False)
        expected_grads = torch.autograd.grad(expected.sum(), ref_inputs)
        actual = compiled(q, k, v, block_mask=block_mask)
        actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            actual_grads,
            expected_grads,
            atol=8e-2,
            rtol=8e-2,
        )

if __name__ == "__main__":
    unittest.main()
