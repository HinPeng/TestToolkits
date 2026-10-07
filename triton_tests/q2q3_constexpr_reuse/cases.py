"""Reviewed source-use expectations, independent of the analyzer's output.

Values describe a legal representative invocation, not tensors to execute.
Unknown expectations explicitly identify a missing proof or unsupported use.
"""

from dataclasses import dataclass, field, replace

RUNTIME = "RuntimeEligible"
SCHEDULE = "ScheduleReusable"
STATIC = "StaticRequired"
UNKNOWN = "Unknown"


@dataclass(frozen=True)
class Expectation:
    classification: str
    why: str
    reason: str = ""


def runtime(why="仅用于整数地址、步长或 mask；没有编译期用途"):
    return Expectation(RUNTIME, why)


def static(why, reason=""):
    return Expectation(STATIC, why, reason)


def unknown(why):
    return Expectation(UNKNOWN, why)


TILE = unknown("R3 形状参数；没有适用于该分块的完整覆盖、启动适配和数值证明，不能仅凭 arange 判 StaticRequired")
CONTROL = static("R1 编译期模式选择；两个分支均需检查", "CONTROL_SPECIALIZATION_PRESERVED")
ASSERT = static("R5 tl.static_assert 的条件必须编译期求值", "STATIC_STATIC_ASSERT")
INCOMPLETE_TUPLE = unknown("当前 DSL 不支持元组解包赋值，使用链不完整，应保守保留")


@dataclass(frozen=True)
class Case:
    id: str
    collection: str
    path: str
    kernel: str
    arguments: dict
    expected: dict
    kind: str = "kernel"
    # Known discrepancies are specific to parameter + observed class + reason.
    # They are populated explicitly below; never learned from analysis results.
    gaps: dict = field(default_factory=dict)


def pointers(names, dtype="float32"):
    return {name: {"pointer": dtype} for name in names.split()}


CASES = [
    Case("q2_store_lowrank", "Q2TritonKernel", "store_lowrank.py", "_store_label_cache_triton_kernel", {
        **pointers("label_cache_ptr key_lr_ptr"),
        **pointers("block_idx_list_ptr token_idx_list_ptr", "int32"),
        "head_num": 8, "head_dim": 64, "token_num": 37,
        "l_stride_b": 8192, "l_stride_h": 1024, "l_stride_t": 64, "l_stride_d": 1,
        "k_stride_s": 512, "k_stride_h": 64, "k_stride_d": 1, "BATCH_BLOCK_NUM": 16,
    }, {
        "head_num": static("真实 head 数，完整 arange 范围随值改变"),
        "head_dim": static("真实 head 维度，完整 arange 范围随值改变"),
        **{name: runtime() for name in ("token_num", "l_stride_b", "l_stride_h", "l_stride_t", "l_stride_d", "k_stride_s", "k_stride_h", "k_stride_d")},
        "BATCH_BLOCK_NUM": TILE,
    }),
    Case("q2_paged_kv", "Q2TritonKernel", "store_paged_kv_cache.py", "_store_paged_kv_cache_kernel", {
        **pointers("k_ptr v_ptr key_cache_ptr value_cache_ptr"),
        **pointers("block_table_ptr cu_seqlens_ptr kv_lens_ptr", "int32"),
        "batch_size": 3, "stride_k_tok": 512, "stride_k_head": 64, "stride_k_dim": 1,
        "stride_v_tok": 512, "stride_v_head": 64, "stride_v_dim": 1,
        "stride_kc_blk": 8192, "stride_kc_head": 1024, "stride_kc_tok": 64, "stride_kc_dim": 1,
        "stride_vc_blk": 8192, "stride_vc_head": 1024, "stride_vc_tok": 64, "stride_vc_dim": 1,
        "stride_bt_batch": 8, "stride_bt_blk": 1, "num_kv_heads": 8,
        "head_dim": 64, "block_size": 16, "CHUNK_SIZE": 32, "IS_DECODE": True, "HAS_KV_LENS": True,
    }, {"head_dim": static("无 head 维度 mask 的完整列范围，决定真实计算维度"),
        "block_size": runtime("整数整除/取余计算物理 KV block 地址；并非 arange 的 tile"),
        "CHUNK_SIZE": TILE, "IS_DECODE": CONTROL, "HAS_KV_LENS": CONTROL}),
    Case("q2_attention_preprocess", "Q2TritonKernel", "dllm_attention/bwd_d.py", "kernel_da_bwd_d", {
        **pointers("fp32o do d"), "TOTAL_S": 129, "N": 4, "H": 64,
        "STRIDE_O_S": 256, "STRIDE_O_N": 64, "STRIDE_O_H": 1,
        "STRIDE_D_S": 1, "STRIDE_D_N": 129, "BLOCK_R": 32,
    }, {"N": runtime(), "H": static("完整 head 维度上的浮点点积归约，改变 H 改变结果"),
        "STRIDE_O_S": runtime(), "STRIDE_O_N": runtime(), "STRIDE_O_H": ASSERT,
        "STRIDE_D_S": ASSERT, "BLOCK_R": TILE}),
    Case("q2_silu", "Q2TritonKernel", "activation/silu.py", "_silu_fwd_kernel", {
        **pointers("x y"), "stride_row": 128, "n_rows": 17, "n_cols": 96,
        "BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 4,
    }, {"BLOCK_SIZE_N": Expectation(SCHEDULE, "T01：完整遍历列轴；与行轴证明组合"),
        "BLOCK_SIZE_M": Expectation(SCHEDULE, "grid-stride 完整覆盖行轴；与固定列遍历及 N 轴证明组合")}),
    Case("q2_swiglu", "Q2TritonKernel", "activation/swiglu.py", "_swiglu_fwd_kernel", {
        **pointers("a b c"), "stride_row": 128, "n_rows": 17, "n_cols": 96,
        "BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 4,
    }, {"BLOCK_SIZE_N": Expectation(SCHEDULE, "T01：完整遍历列轴；与行轴证明组合"),
        "BLOCK_SIZE_M": Expectation(SCHEDULE, "grid-stride 完整覆盖行轴；与固定列遍历及 N 轴证明组合")}),
    Case("q2_rmsnorm", "Q2TritonKernel", "rmsnorm.py", "_rmsnorm_infer_kernel", {
        **pointers("X_ptr Y_ptr W_ptr"), "stride_x_row": 512, "stride_y_row": 512,
        "n_rows": 17, "n_cols": 384, "eps": 1e-5, "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128,
    }, {"BLOCK_SIZE_M": TILE,
        "BLOCK_SIZE_N": static("列 tile 改变块内 sum 与块间累加分组，首版保留浮点归约顺序 (§4.2 S4)")}),
    Case("q2_position_ids", "Q2TritonKernel", "fla/ops/utils/index.py", "prepare_position_ids_kernel", {
        **pointers("y cu_seqlens", "int32"), "B": 128,
    }, {"B": INCOMPLETE_TUPLE}),
    Case("q2_load_predicate", "Q2TritonKernel", "utils.py", "load_with_pred_1d", {
        **pointers("ptr", "int32"), "skip_boundary_check": False, "mask": True, "other": -1,
    }, {"skip_boundary_check": CONTROL}, kind="helper"),
    Case("q2_store_predicate", "Q2TritonKernel", "utils.py", "store_with_pred_1d", {
        **pointers("ptr", "int32"), "value": 7, "skip_boundary_check": False, "mask": True,
    }, {"skip_boundary_check": CONTROL}, kind="helper"),
    Case("q2_rope_extension", "Q2TritonKernel", "rope.py", "_compute_rope", {
        "x": 0.5, "sin_tile": 0.5, "cos_tile": 0.5,
        "head_num": 8, "half_rope_dim": 32, "TOKEN_BLOCK_SIZE": 4, "inverse": False,
    }, {name: unknown("缺少 CANN extract_slice/insert_slice 摘要，未完成后续使用链")
        for name in ("head_num", "half_rope_dim", "TOKEN_BLOCK_SIZE", "inverse")}, kind="helper"),
    Case("q2_rope_separated", "Q2TritonKernel", "rope.py", "_compute_rope_separated", {
        "x1": 0.5, "x2": 0.25, "sin_tile": 0.5, "cos_tile": 0.5, "inverse": False,
    }, {"inverse": CONTROL}, kind="helper"),
    Case("q2_sdpa_dtype", "Q2TritonKernel", "sdpa.py", "kernel_sdpa_bwd_d", {
        **pointers("o do d"), "B": 2, "N": 4, "S": 129, "H": 64,
        "STRIDE_O_B": 33024, "STRIDE_O_N": 8256, "STRIDE_O_S": 64, "STRIDE_O_H": 1,
        "STRIDE_D_B": 516, "STRIDE_D_N": 129, "STRIDE_D_S": 1, "BLOCK_R": 32,
        "LOW_TYPE": {"dtype": "bfloat16"}, "HIGH_TYPE": {"dtype": "float32"},
    }, {
        **{name: runtime() for name in ("B", "N", "STRIDE_O_B", "STRIDE_O_N", "STRIDE_O_S", "STRIDE_O_H", "STRIDE_D_B", "STRIDE_D_N", "STRIDE_D_S")},
        "S": unknown("S 进入 tl.cdiv，当前尚无其动态整数类型证明"),
        "H": static("完整 head 维度上的浮点归约"), "BLOCK_R": TILE,
        "LOW_TYPE": unknown("未使用的 dtype constexpr 不在首版整数动态化支持范围"),
        "HIGH_TYPE": static("R4 传入 .to(HIGH_TYPE) 的 dtype 必须静态", "STATIC_DTYPE"),
    }),
    Case("q2_conv_dh0", "Q2TritonKernel", "fla/modules/causal_conv1d_bwd.py", "compute_dh0_kernel", {
        **pointers("dy y weight dh0"), **pointers("cu_seqlens", "int32"),
        "stride_dy_n": 8192, "stride_dy_t": 64, "T": 128,
        "D": 64, "W": 4, "BD": 32, "USE_ACTIVATION": True, "IS_VARLEN": False,
    }, {name: INCOMPLETE_TUPLE for name in ("D", "W", "BD", "USE_ACTIVATION", "IS_VARLEN")}),
    Case("q3_flat_offset", "Q3TritonKernel", "fla/modules/activations.py", "_flat_offset", {
        "offs": 17, "D": 32, "stride": 64, "IS_LINEAR": True,
    }, {"D": runtime("即使 IS_LINEAR=True，仍检查另一分支中的整除/取余地址链"),
        "IS_LINEAR": CONTROL}, kind="helper"),
    Case("q3_position_embeddings", "Q3TritonKernel", "recsys_example/add_position_embeddings/add_position_embeddings.py", "_add_position_embeddings_kernel", {
        **pointers("Jagged Dense Out"), **pointers("seq_offsets high_inds ind_offsets", "int32"),
        "AUTOTUNE_MAX_SEQ_LEN": 128, "D": 96, "scale": 0.5,
        "stride_jn": 96, "stride_dk": 96, "stride_on": 96,
        "SCALE_JAGGED": True, "BLOCK_D": 64, "BLOCK_N": 32,
    }, {"SCALE_JAGGED": CONTROL, "BLOCK_D": TILE, "BLOCK_N": TILE}),
    Case("q3_jagged_to_dense", "Q3TritonKernel", "fbgemm/triton_jagged_to_dense/triton_jagged_to_dense.py", "triton_jagged_to_dense", {
        **pointers("jagged_value_ptr output_dense_ptr operation_dense"),
        **pointers("jagged_offsets_ptr dense_indices_ptr", "int32"),
        "jagged_value_row_stride": 64, "dense_col_stride": 1, "dense_row_stride": 64, "dense_matrix_stride": 4096,
        "JAGGED_DIM": 2, "thread_block_row_size": 32, "thread_block_col_size": 32, "operation_function": "add",
    }, {"JAGGED_DIM": CONTROL, "thread_block_row_size": TILE, "thread_block_col_size": TILE,
        "operation_function": static("字符串 constexpr 选择 add/mul/无融合分支", "CONTROL_SPECIALIZATION_PRESERVED")}),
    Case("q3_silu", "Q3TritonKernel", "recsys_example/silu/silu.py", "_silu_forward", {
        **pointers("output_ptr input_ptr"), "x_size": 10001, "x_block_size": 8192,
    }, {"x_block_size": TILE}),
    Case("q3_layernorm", "Q3TritonKernel", "recsys_example/layer_norm/layer_norm.py", "_layer_norm_fwd", {
        **pointers("X Y Mean Rstd"), "D": 96, "eps": 1e-5, "stride_x": 128, "stride_y": 128,
        "TRAINING": True, "BLOCK_D": 128, "COMPUTE_MEAN_AND_RSTD": True,
    }, {"TRAINING": unknown("分析在浮点除法处中断，尚未读到 TRAINING 分支"),
        "BLOCK_D": static("改变块内浮点归约树；首版保留数值顺序"), "COMPUTE_MEAN_AND_RSTD": CONTROL}),
    Case("q3_l2norm", "Q3TritonKernel", "fla/modules/l2norm.py", "l2norm_fwd_kernel1", {
        **pointers("x y rstd"), "eps": 1e-5, "D": 96, "BD": 128, "T": 1024, "BR": 4,
    }, {"D": unknown("sqrt 无摘要，无法完成全部地址/mask 使用链"),
        "BD": static("改变浮点 sum 归约树，首版保留数值顺序"),
        "T": unknown("前一分支 sqrt 无摘要，尚未读到后一分支 T 的 static_assert"), "BR": CONTROL}),
    Case("q3_topk_partial", "Q3TritonKernel", "minimax_m3/topk_index_partial_kernel.py", "_topk_index_partial_kernel", {
        **pointers("s_ptr ts_partial_ptr"), **pointers("ti_partial_ptr seq_lens", "int32"),
        "block_size": 128, "topk": 16, "chunk_blocks": 64, "decode_query_len": 2,
        "stride_s_h": 1024, "stride_s_b": 4096, "stride_s_k": 1,
        "stride_ts_c": 128, "stride_ts_h": 32, "stride_ts_b": 16, "stride_ts_t": 1,
        "stride_ti_c": 128, "stride_ti_h": 32, "stride_ti_b": 16, "stride_ti_t": 1,
        "BLOCK_SIZE_K": 64, "BLOCK_SIZE_T": 16,
    }, {"block_size": INCOMPLETE_TUPLE, "topk": ASSERT, "BLOCK_SIZE_K": ASSERT,
        "BLOCK_SIZE_T": static("决定无 mask 输出槽数量与 top-k 归并结果布局；不是已获准的逐元素调度 tile")}),
    Case("q3_pack", "Q3TritonKernel", "fla/ops/utils/pack.py", "packunpack_sequence_kernel", {
        **pointers("x y"), **pointers("cu_seqlens", "int32"), "S": 128, "D": 64,
        "BD": 64, "PADDING_SIDE": "left", "PACK": True,
    }, {name: INCOMPLETE_TUPLE for name in ("BD", "PADDING_SIDE", "PACK")}),
    Case("q3_pooling", "Q3TritonKernel", "fla/ops/utils/pooling.py", "mean_pooling_fwd_kernel", {
        **pointers("x o"), **pointers("cu_seqlens chunk_indices", "int32"),
        "T": 129, "H": 4, "D": 64, "BT": 32, "BD": 32, "IS_VARLEN": False,
    }, {name: INCOMPLETE_TUPLE for name in ("H", "D", "BT", "BD", "IS_VARLEN")}),
]


@dataclass(frozen=True)
class Gap:
    issue: str
    observed: str
    reason: str


TILE_GAP = Gap("G1_SHAPE_IS_NOT_STATIC_PROOF", STATIC, "STATIC_ARANGE")
POINTER_GAP = Gap("G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC", UNKNOWN, "FLOAT_USE_NOT_SUPPORTED")

tile_gaps = {
    "q2_store_lowrank": "BATCH_BLOCK_NUM", "q2_paged_kv": "CHUNK_SIZE",
    "q2_attention_preprocess": "BLOCK_R", "q2_silu": "",
    "q2_swiglu": "", "q2_rmsnorm": "BLOCK_SIZE_M",
    "q2_sdpa_dtype": "BLOCK_R", "q3_position_embeddings": "BLOCK_D BLOCK_N",
    "q3_jagged_to_dense": "thread_block_row_size thread_block_col_size", "q3_silu": "x_block_size",
}
pointer_gaps = {
    "q2_store_lowrank": "l_stride_b l_stride_h l_stride_t l_stride_d k_stride_s k_stride_h k_stride_d",
    "q2_paged_kv": "block_size", "q2_attention_preprocess": "N STRIDE_O_S STRIDE_O_N",
    "q2_sdpa_dtype": "B N STRIDE_O_B STRIDE_O_N STRIDE_O_S STRIDE_O_H STRIDE_D_B STRIDE_D_N STRIDE_D_S",
}
CASES = [replace(case, gaps={
    **{name: TILE_GAP for name in tile_gaps.get(case.id, "").split()},
    **{name: POINTER_GAP for name in pointer_gaps.get(case.id, "").split()},
}) for case in CASES]


# Alternate values must not hide static uses by pruning a currently untaken
# branch. Boundary token counts and strides exercise the same type context.
for case_id, suffix, changes in (
    ("q2_store_lowrank", "tail", {"token_num": 1, "l_stride_d": 2, "k_stride_d": 2}),
    ("q2_paged_kv", "prefill", {"IS_DECODE": False, "HAS_KV_LENS": False}),
    ("q2_load_predicate", "unchecked", {"skip_boundary_check": True}),
    ("q2_rope_separated", "inverse", {"inverse": True}),
    ("q3_flat_offset", "strided", {"IS_LINEAR": False, "D": 64}),
    ("q3_position_embeddings", "unscaled", {"SCALE_JAGGED": False}),
    ("q3_jagged_to_dense", "no_fusion", {"JAGGED_DIM": 3, "operation_function": None}),
    ("q3_l2norm", "single_row", {"BR": 1, "T": 0}),
):
    base = next(case for case in CASES if case.id == case_id)
    CASES.append(replace(base, id=base.id + "_" + suffix, arguments={**base.arguments, **changes}))

# A type-control profile, explicitly distinguished from the normal FP32 case.
# Integer address uses should not become float arithmetic when pointee changes.
for case_id, names in (("q2_store_lowrank", "label_cache_ptr key_lr_ptr"),
                       ("q2_paged_kv", "k_ptr v_ptr key_cache_ptr value_cache_ptr")):
    base = next(case for case in CASES if case.id == case_id)
    CASES.append(replace(base, id=base.id + "_int32", arguments={**base.arguments, **pointers(names, "int32")},
                         gaps={name: gap for name, gap in base.gaps.items() if gap is TILE_GAP}))

CASES = tuple(CASES)
