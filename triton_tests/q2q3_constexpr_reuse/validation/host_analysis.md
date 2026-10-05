# Q2/Q3 constexpr 纯分析验证

22 个 JIT 定义，32 个输入 profile，132 项参数断言：82 符合预期，50 不符。

使用当前 checkout 的分析器；没有执行 kernel、编译产物或验证 NPU 数值/ABI。
分析器版本：`3105ad280023074be867d748b20b61f3d3eef950`，规则版本 3。
源码及实现 SHA-256、工作区改动、完整绑定和分类理由见同名 JSON。

`G1`：R3 形状用途不能独立证明 StaticRequired；缺少调度证明应为 Unknown。
`G2`：浮点 pointee 的地址计算属于整数/指针运算，不能按浮点数据算术拒绝。

## 分类差异

| Profile | 参数 | 预期 | 实际 | 差异 |
| --- | --- | --- | --- | --- |
| q2_store_lowrank | l_stride_b | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | l_stride_h | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | l_stride_t | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | l_stride_d | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | k_stride_s | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | k_stride_h | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | k_stride_d | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank | BATCH_BLOCK_NUM | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_paged_kv | block_size | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_paged_kv | CHUNK_SIZE | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_attention_preprocess | N | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_attention_preprocess | STRIDE_O_S | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_attention_preprocess | STRIDE_O_N | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_attention_preprocess | BLOCK_R | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_silu | BLOCK_SIZE_N | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_silu | BLOCK_SIZE_M | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_swiglu | BLOCK_SIZE_N | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_swiglu | BLOCK_SIZE_M | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_rmsnorm | BLOCK_SIZE_M | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_sdpa_dtype | B | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | N | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_O_B | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_O_N | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_O_S | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_O_H | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_D_B | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_D_N | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | STRIDE_D_S | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_sdpa_dtype | BLOCK_R | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_position_embeddings | BLOCK_D | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_position_embeddings | BLOCK_N | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_jagged_to_dense | thread_block_row_size | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_jagged_to_dense | thread_block_col_size | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_silu | x_block_size | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_store_lowrank_tail | l_stride_b | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | l_stride_h | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | l_stride_t | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | l_stride_d | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | k_stride_s | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | k_stride_h | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | k_stride_d | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_store_lowrank_tail | BATCH_BLOCK_NUM | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_paged_kv_prefill | block_size | RuntimeEligible | Unknown | G2_FLOAT_POINTER_IS_NOT_FLOAT_ARITHMETIC |
| q2_paged_kv_prefill | CHUNK_SIZE | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_position_embeddings_unscaled | BLOCK_D | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_position_embeddings_unscaled | BLOCK_N | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_jagged_to_dense_no_fusion | thread_block_row_size | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q3_jagged_to_dense_no_fusion | thread_block_col_size | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_store_lowrank_int32 | BATCH_BLOCK_NUM | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |
| q2_paged_kv_int32 | CHUNK_SIZE | Unknown | StaticRequired | G1_SHAPE_IS_NOT_STATIC_PROOF |

## 全部用例

| Profile | 来源（定义行） | 符合/总数 |
| --- | --- | --- |
| q2_store_lowrank | Q2TritonKernel/src/kernels/store_lowrank.py:11 `_store_label_cache_triton_kernel` | 3/11 |
| q2_paged_kv | Q2TritonKernel/src/kernels/store_paged_kv_cache.py:9 `_store_paged_kv_cache_kernel` | 3/5 |
| q2_attention_preprocess | Q2TritonKernel/src/kernels/dllm_attention/bwd_d.py:14 `kernel_da_bwd_d` | 3/7 |
| q2_silu | Q2TritonKernel/src/kernels/activation/silu.py:35 `_silu_fwd_kernel` | 0/2 |
| q2_swiglu | Q2TritonKernel/src/kernels/activation/swiglu.py:37 `_swiglu_fwd_kernel` | 0/2 |
| q2_rmsnorm | Q2TritonKernel/src/kernels/rmsnorm.py:49 `_rmsnorm_infer_kernel` | 1/2 |
| q2_position_ids | Q2TritonKernel/src/kernels/fla/ops/utils/index.py:25 `prepare_position_ids_kernel` | 1/1 |
| q2_load_predicate | Q2TritonKernel/src/kernels/utils.py:53 `load_with_pred_1d` | 1/1 |
| q2_store_predicate | Q2TritonKernel/src/kernels/utils.py:61 `store_with_pred_1d` | 1/1 |
| q2_rope_extension | Q2TritonKernel/src/kernels/rope.py:71 `_compute_rope` | 4/4 |
| q2_rope_separated | Q2TritonKernel/src/kernels/rope.py:103 `_compute_rope_separated` | 1/1 |
| q2_sdpa_dtype | Q2TritonKernel/src/kernels/sdpa.py:440 `kernel_sdpa_bwd_d` | 4/14 |
| q2_conv_dh0 | Q2TritonKernel/src/kernels/fla/modules/causal_conv1d_bwd.py:349 `compute_dh0_kernel` | 5/5 |
| q3_flat_offset | Q3TritonKernel/src/kernels/fla/modules/activations.py:101 `_flat_offset` | 2/2 |
| q3_position_embeddings | Q3TritonKernel/src/kernels/recsys_example/add_position_embeddings/add_position_embeddings.py:53 `_add_position_embeddings_kernel` | 1/3 |
| q3_jagged_to_dense | Q3TritonKernel/src/kernels/fbgemm/triton_jagged_to_dense/triton_jagged_to_dense.py:71 `triton_jagged_to_dense` | 2/4 |
| q3_silu | Q3TritonKernel/src/kernels/recsys_example/silu/silu.py:29 `_silu_forward` | 0/1 |
| q3_layernorm | Q3TritonKernel/src/kernels/recsys_example/layer_norm/layer_norm.py:37 `_layer_norm_fwd` | 3/3 |
| q3_l2norm | Q3TritonKernel/src/kernels/fla/modules/l2norm.py:42 `l2norm_fwd_kernel1` | 4/4 |
| q3_topk_partial | Q3TritonKernel/src/kernels/minimax_m3/topk_index_partial_kernel.py:43 `_topk_index_partial_kernel` | 4/4 |
| q3_pack | Q3TritonKernel/src/kernels/fla/ops/utils/pack.py:29 `packunpack_sequence_kernel` | 3/3 |
| q3_pooling | Q3TritonKernel/src/kernels/fla/ops/utils/pooling.py:30 `mean_pooling_fwd_kernel` | 5/5 |
| q2_store_lowrank_tail | Q2TritonKernel/src/kernels/store_lowrank.py:11 `_store_label_cache_triton_kernel` | 3/11 |
| q2_paged_kv_prefill | Q2TritonKernel/src/kernels/store_paged_kv_cache.py:9 `_store_paged_kv_cache_kernel` | 3/5 |
| q2_load_predicate_unchecked | Q2TritonKernel/src/kernels/utils.py:53 `load_with_pred_1d` | 1/1 |
| q2_rope_separated_inverse | Q2TritonKernel/src/kernels/rope.py:103 `_compute_rope_separated` | 1/1 |
| q3_flat_offset_strided | Q3TritonKernel/src/kernels/fla/modules/activations.py:101 `_flat_offset` | 2/2 |
| q3_position_embeddings_unscaled | Q3TritonKernel/src/kernels/recsys_example/add_position_embeddings/add_position_embeddings.py:53 `_add_position_embeddings_kernel` | 1/3 |
| q3_jagged_to_dense_no_fusion | Q3TritonKernel/src/kernels/fbgemm/triton_jagged_to_dense/triton_jagged_to_dense.py:71 `triton_jagged_to_dense` | 2/4 |
| q3_l2norm_single_row | Q3TritonKernel/src/kernels/fla/modules/l2norm.py:42 `l2norm_fwd_kernel1` | 4/4 |
| q2_store_lowrank_int32 | Q2TritonKernel/src/kernels/store_lowrank.py:11 `_store_label_cache_triton_kernel` | 10/11 |
| q2_paged_kv_int32 | Q2TritonKernel/src/kernels/store_paged_kv_cache.py:9 `_store_paged_kv_cache_kernel` | 4/5 |

理由行号为分析器原始 JIT 相对行号；helper 理由可能相对 helper，不能直接加到入口定义行。
