# Frozen Q2/Q3 JIT definitions; provenance and hashes are in sources.json.
# Function text and JIT decorators are unchanged. Host wrappers and non-JIT
# decorators (autotune, heuristics, libentry) are omitted for source analysis.

import triton
import triton.language as tl


# Source: Q2TritonKernel/src/kernels/store_lowrank.py:11
@triton.jit
def _store_label_cache_triton_kernel(
    label_cache_ptr,
    key_lr_ptr,
    block_idx_list_ptr,
    token_idx_list_ptr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    token_num: tl.constexpr,
    l_stride_b: tl.constexpr,
    l_stride_h: tl.constexpr,
    l_stride_t: tl.constexpr,
    l_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    BATCH_BLOCK_NUM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_start = pid_b * BATCH_BLOCK_NUM
    b_end = tl.minimum(b_start + BATCH_BLOCK_NUM, token_num)
    b = tl.arange(0, BATCH_BLOCK_NUM) + b_start
    b_3d = b[:, None, None]
    h = tl.arange(0, head_num)
    h_3d = h[None, :, None]
    d = tl.arange(0, head_dim)
    d_3d = d[None, None, :]
    block_idx = tl.load(block_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)
    token_idx = tl.load(token_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)

    label_cache_addr = block_idx * l_stride_b + h_3d * l_stride_h + token_idx * l_stride_t + d_3d * l_stride_d
    key_lr_offset = b_3d * k_stride_s + h_3d * k_stride_h + d_3d * k_stride_d

    valid_mask = (b_3d < b_end) & (h_3d < head_num) & (d_3d < head_dim)

    key_lr_data = tl.load(key_lr_ptr + key_lr_offset, mask=valid_mask, other=0.0)
    tl.store(label_cache_ptr + label_cache_addr, key_lr_data, mask=valid_mask)


# Source: Q2TritonKernel/src/kernels/store_paged_kv_cache.py:9
@triton.jit
def _store_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    kv_lens_ptr,
    batch_size,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_kc_blk,
    stride_kc_head,
    stride_kc_tok,
    stride_kc_dim,
    stride_vc_blk,
    stride_vc_head,
    stride_vc_tok,
    stride_vc_dim,
    stride_bt_batch,
    stride_bt_blk,
    num_kv_heads,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    prev_chunks = 0
    for batch_idx in range(batch_size):
        if IS_DECODE:
            seq_start_tok = batch_idx
            seq_len_curr = 1
        else:
            seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
            seq_end_tok = tl.load(cu_seqlens_ptr + batch_idx + 1)
            seq_len_curr = seq_end_tok - seq_start_tok

        write_start = 0
        if HAS_KV_LENS:
            write_start = tl.load(kv_lens_ptr + batch_idx)
        valid_write = write_start >= 0
        cur_chunks = tl.where(valid_write, tl.cdiv(seq_len_curr, CHUNK_SIZE), 0)
        start_chunk = (pid + num_programs - prev_chunks % num_programs) % num_programs
        prev_chunks += cur_chunks

        for chunk_idx in range(start_chunk, cur_chunks, num_programs):
            token_offset_in_seq = chunk_idx * CHUNK_SIZE
            valid_len = seq_len_curr - token_offset_in_seq
            curr_log_pos = write_start + token_offset_in_seq
            curr_kv_pos = seq_start_tok + token_offset_in_seq

            remain_chunk_len = tl.minimum(CHUNK_SIZE, valid_len)

            processed = 0
            while processed < remain_chunk_len:
                block_table_idx = curr_log_pos // block_size
                block_inner_off = curr_log_pos % block_size

                physical_block_id = tl.load(
                    block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk
                )
                valid_block = physical_block_id >= 0
                physical_block_id = tl.maximum(physical_block_id, 0)

                space_in_block = block_size - block_inner_off
                sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

                offs_sub = tl.arange(0, CHUNK_SIZE)
                mask_sub = offs_sub < sub_len

                offs_d = tl.arange(0, head_dim)

                for h in range(num_kv_heads):
                    src_k_ptr = (
                        k_ptr
                        + (curr_kv_pos + offs_sub[:, None]) * stride_k_tok
                        + h * stride_k_head
                        + offs_d[None, :] * stride_k_dim
                    )

                    k_val = tl.load(src_k_ptr, mask=mask_sub[:, None], other=0.0)

                    dst_k_ptr = (
                        key_cache_ptr
                        + physical_block_id * stride_kc_blk
                        + h * stride_kc_head
                        + (block_inner_off + offs_sub[:, None]) * stride_kc_tok
                        + offs_d[None, :] * stride_kc_dim
                    )

                    tl.store(dst_k_ptr, k_val, mask=valid_block & mask_sub[:, None])

                    src_v_ptr = (
                        v_ptr
                        + (curr_kv_pos + offs_sub[:, None]) * stride_v_tok
                        + h * stride_v_head
                        + offs_d[None, :] * stride_v_dim
                    )

                    v_val = tl.load(src_v_ptr, mask=mask_sub[:, None], other=0.0)

                    dst_v_ptr = (
                        value_cache_ptr
                        + physical_block_id * stride_vc_blk
                        + h * stride_vc_head
                        + (block_inner_off + offs_sub[:, None]) * stride_vc_tok
                        + offs_d[None, :] * stride_vc_dim
                    )

                    tl.store(dst_v_ptr, v_val, mask=valid_block & mask_sub[:, None])

                processed += sub_len
                curr_log_pos += sub_len
                curr_kv_pos += sub_len


# Source: Q2TritonKernel/src/kernels/dllm_attention/bwd_d.py:19
@triton.jit(do_not_specialize=["TOTAL_S", "STRIDE_D_N"])
def kernel_da_bwd_d(
    fp32o,
    do,
    d,
    TOTAL_S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N,
    BLOCK_R: tl.constexpr,
):
    tl.static_assert(STRIDE_O_H == 1)
    tl.static_assert(STRIDE_D_S == 1)
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(TOTAL_S, BLOCK_R)
    for task_id in range(pid, num_r * N, tl.num_programs(axis=0)):
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        offs_h = tl.arange(0, H)
        ptr_fp32o = (
            fp32o
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + offs_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + offs_h[None, :] * STRIDE_O_H
        )
        ptr_d = d + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        mask_o = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < TOTAL_S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < TOTAL_S

        block_o = tl.load(ptr_fp32o, mask=mask_o, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_o, other=0.0)
        block_d = tl.sum(block_do.to(tl.float32) * block_o, axis=1)
        tl.store(ptr_d, block_d, mask=mask_d)


# Source: Q2TritonKernel/src/kernels/activation/silu.py:16
@triton.jit
def silu_activation(x):
    return x * tl.sigmoid(x)


# Source: Q2TritonKernel/src/kernels/activation/silu.py:26
@triton.jit
def _silu_fwd_kernel(
    x,
    y,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            y_ptrs = y + rows_off[:, None] * stride_row + cols_off[None, :]

            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            y_f32 = silu_activation(x_f32)

            y_chunk = y_f32.to(x_chunk.dtype)

            tl.store(y_ptrs, y_chunk, mask=block_mask)


# Source: Q2TritonKernel/src/kernels/activation/swiglu.py:18
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


# Source: Q2TritonKernel/src/kernels/activation/swiglu.py:28
@triton.jit
def _swiglu_fwd_kernel(
    a,
    b,
    c,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            c_ptrs = c + rows_off[:, None] * stride_row + cols_off[None, :]

            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            silu_a = silu(a_f32)

            c_chunk = silu_a.to(a_chunk.dtype) * b_chunk

            tl.store(c_ptrs, c_chunk, mask=block_mask)


# Source: Q2TritonKernel/src/kernels/rmsnorm.py:53
@triton.jit
def _rmsnorm_infer_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < n_rows

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

            ss_acc += tl.sum(x * x, axis=1)

        ss_acc = tl.where(row_mask, ss_acc, 0)

        mean_square = ss_acc / n_cols
        rrms = tl.rsqrt(mean_square + eps)

        rrms = tl.where(row_mask, rrms, 0.0)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
            w_ptrs = W_ptr + col_offsets
            y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=col_mask, other=0.0)

            x_f32 = x.to(tl.float32)
            w_f32 = w.to(tl.float32)

            x_normalized = x_f32 * rrms[:, None]

            y = x_normalized * w_f32[None, :]

            tl.store(
                y_ptrs,
                y.to(Y_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )


# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Source: Q2TritonKernel/src/kernels/fla/ops/utils/index.py:25
@triton.jit
def prepare_position_ids_kernel(
    y,
    cu_seqlens,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


# Source: Q2TritonKernel/src/kernels/utils.py:53
@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)


# Source: Q2TritonKernel/src/kernels/utils.py:61
@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask: tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)


# Source: Q2TritonKernel/src/kernels/rope.py:71
@triton.jit
def _compute_rope(
    x,
    sin_tile,
    cos_tile,
    head_num: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    inverse: tl.constexpr,
):
    x1 = tl.extra.cann.extension.extract_slice(x, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x2 = tl.extra.cann.extension.extract_slice(x, [0, 0, half_rope_dim], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])

    if inverse:
        roped_x1 = x1 * cos_tile + x2 * sin_tile
        roped_x2 = x2 * cos_tile - x1 * sin_tile
    else:
        roped_x1 = x1 * cos_tile - x2 * sin_tile
        roped_x2 = x2 * cos_tile + x1 * sin_tile

    x = tl.extra.cann.extension.insert_slice(x, roped_x1, [0, 0, 0], [TOKEN_BLOCK_SIZE, head_num, half_rope_dim], [1, 1, 1])
    x = tl.extra.cann.extension.insert_slice(
        x,
        roped_x2,
        [0, 0, half_rope_dim],
        [TOKEN_BLOCK_SIZE, head_num, half_rope_dim],
        [1, 1, 1],
    )

    return x


# Source: Q2TritonKernel/src/kernels/rope.py:103
@triton.jit
def _compute_rope_separated(
    x1,
    x2,
    sin_tile,
    cos_tile,
    inverse: tl.constexpr,
):
    if inverse:
        roped_x1 = x1 * cos_tile + x2 * sin_tile
        roped_x2 = x2 * cos_tile - x1 * sin_tile
    else:
        roped_x1 = x1 * cos_tile - x2 * sin_tile
        roped_x2 = x2 * cos_tile + x1 * sin_tile
    return roped_x1, roped_x2


# Source: Q2TritonKernel/src/kernels/sdpa.py:442
@triton.jit
def kernel_sdpa_bwd_d(
    o,
    do,
    d,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    for task_id in range(pid, num_r * B * N, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)
        ptr_o = (
            o
            + idx_b * STRIDE_O_B
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do
            + idx_b * STRIDE_O_B
            + idx_n * STRIDE_O_N
            + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S
            + idx_h[None, :] * STRIDE_O_H
        )
        ptr_d = d + idx_b * STRIDE_D_B + idx_n * STRIDE_D_N + (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        mask_o = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S
        mask_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S

        block_o = tl.load(ptr_o, mask=mask_o, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_o, other=0.0)
        block_d = tl.sum(block_do.to(HIGH_TYPE) * block_o.to(HIGH_TYPE), axis=1)
        tl.store(ptr_d, block_d, mask=mask_d)


# Source: Q2TritonKernel/src/kernels/fla/modules/causal_conv1d_bwd.py:326
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        dy_base = dy + tl.cast(i_n, tl.int64) * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t

                p_dy = dy_base + t * stride_dy_t + o_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_dy_t + t * stride_dy_t + o_d
                    else:
                        p_y = y + tl.cast(i_n, tl.int64) * stride_dy_n + t * stride_dy_t + o_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)

                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Source: Q3TritonKernel/src/kernels/fla/modules/activations.py:101
@triton.jit
def _flat_offset(
    offs,
    D: tl.constexpr,
    stride,
    IS_LINEAR: tl.constexpr,
):
    if IS_LINEAR:
        return offs
    row = offs // D
    col = offs % D
    return row * stride + col


# Migrated from recsys-examples/examples/hstu/ops/triton_ops/triton_position.py
# Only the non-timestamp position embeddings (forward + bwd) is migrated.
# Changes:
#   - Imports updated to src.kernels.recsys_example.utils
#   - torch.library.wrap_triton(...) calls replaced with direct kernel[(grid,)](...) calls


# Source: Q3TritonKernel/src/kernels/recsys_example/add_position_embeddings/add_position_embeddings.py:53
@triton.jit
def _add_position_embeddings_kernel(
    Jagged,
    seq_offsets,
    high_inds,
    Dense,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    scale,
    ind_offsets,
    stride_jn,
    stride_dk,
    stride_on,
    SCALE_JAGGED: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Jagged has shape (sum_B(N_i), D),
    Dense has shape (K, D),
    Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    max_ind = tl.load(high_inds + off_b)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    if ind_offsets is None:
        clamped_offs_n = tl.where(offs_n >= max_ind, max_ind, offs_n)
    else:
        ind_offset = tl.load(ind_offsets + off_b)
        clamped_offs_n = tl.where(
            offs_n + ind_offset >= max_ind, max_ind, offs_n + ind_offset
        )
    offs_d = tl.arange(0, BLOCK_D)
    Jagged += seq_start.to(tl.int64) * stride_jn
    jagged_ptr_offsets = offs_n[:, None] * stride_jn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_ptrs = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    dense_ptrs = Dense + clamped_offs_n[:, None] * stride_dk + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < D)
        jg = tl.load(Jagged + jagged_ptr_offsets, mask=mask)
        if SCALE_JAGGED:
            jg = jg * scale
        dn = tl.load(dense_ptrs, mask=mask)
        jg += dn
        tl.store(out_ptrs, jg, mask=mask)
        dense_ptrs += BLOCK_D
        out_ptrs += BLOCK_D
        offs_d += BLOCK_D
        jagged_ptr_offsets += BLOCK_D


#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NPU migration of `triton_jagged_to_dense` (and the `jagged_to_dense` Python
# wrapper) from
#   fbgemm_gpu/fbgemm_gpu/triton/jagged/triton_jagged_tensor_ops.py
#
# Migration notes (see migrate_from_gpu.md):
#   * added `import torch_npu` for Ascend NPU runtime support;
#   * replaced `device="cuda"` / `.cuda()` with `device="npu"` / `.npu()`;
#   * the Triton kernel body itself is kept unchanged, only the host-side
#     device handling is migrated.
#
# Unlike the upstream wrapper (which dispatches the 2D case to a separate
# `triton_jagged_to_dense_optimization_2d` kernel), this file exercises the
# target kernel `triton_jagged_to_dense` directly for every jagged dimension.
# To make that correct for the 2D case as well, the output dense tensor is
# initialised with `padding_value` (via `torch.full`) so the positions that the
# kernel skips (out-of-range) retain the padded value.


# Source: Q3TritonKernel/src/kernels/fbgemm/triton_jagged_to_dense/triton_jagged_to_dense.py:35
@triton.jit
def tensor_elementwise_add(x, y):
    return x + y


# Source: Q3TritonKernel/src/kernels/fbgemm/triton_jagged_to_dense/triton_jagged_to_dense.py:40
@triton.jit
def tensor_elementwise_mul(x, y):
    return x * y


# Source: Q3TritonKernel/src/kernels/fbgemm/triton_jagged_to_dense/triton_jagged_to_dense.py:71
@triton.jit
def triton_jagged_to_dense(
    # only constexpr annotations support in triton now
    jagged_value_ptr,
    jagged_offsets_ptr,
    jagged_value_row_stride,
    output_dense_ptr,
    dense_indices_ptr,
    dense_col_stride,  # stride of output dense with dimension (z,y,x)
    dense_row_stride,
    dense_matrix_stride,
    JAGGED_DIM: tl.constexpr,  # number of dimension of jagged tensor
    thread_block_row_size: tl.constexpr,
    thread_block_col_size: tl.constexpr,
    operation_function: tl.constexpr,  # fusion arithmetic operation function and it's input dense
    operation_dense,
) -> None:
    pid = tl.program_id(0)

    # begin index and end index of jagged tensor Values
    begin = tl.load(jagged_offsets_ptr + pid)
    end = tl.load(jagged_offsets_ptr + (pid + 1))

    # adjust the address of the jagged tensor Values to the correct address
    jagged_value_ptr += begin * jagged_value_row_stride

    # if it's 2D (or 1D) Jagged tensor we can direct use the offset in offsets
    # ( since there is only one offset ) else we actually need to use the
    # preprocess index to found the correct address of dense
    if JAGGED_DIM > 2:
        # read the index for current kernel
        dense_indice = tl.load(dense_indices_ptr + pid)

        # if the dense_indice is -1 which mean it's a truncation case
        # in that case we don't need to do anything since the dense
        # initialize with padded value
        if dense_indice == -1:
            return

        # adjust the address of output dense ptr to the correct address
        output_dense_ptr += dense_indice

        # also need to update the operation function if exist
        # notice dense_indice of two is same because we assume
        # the two dense + dense are same size
        if operation_function is not None:
            operation_dense += dense_indice
    else:
        output_dense_ptr += pid * dense_matrix_stride

        if operation_function is not None:
            operation_dense += pid * dense_matrix_stride

    offset_row = tl.arange(0, thread_block_row_size)

    # boundary need for the mask since it could be dense's size smaller than
    # jagged tensor or revert case
    N = tl.minimum(dense_row_stride, jagged_value_row_stride)
    M = tl.minimum(dense_matrix_stride // dense_row_stride, end - begin)

    for _i in range(begin, end, thread_block_row_size):
        offset_col = tl.arange(0, thread_block_col_size)
        block_offset = (
            offset_row[:, None] * dense_row_stride
            + offset_col[None, :] * dense_col_stride
        )
        for _j in range(0, N, thread_block_col_size):
            mask = (offset_row[:, None] < M) & (offset_col[None, :] < N)
            jagged_val = tl.load(jagged_value_ptr + block_offset, mask=mask, other=0)

            # if there is some arithmetic operation we do the fusion computation
            if operation_function is not None:
                val1 = jagged_val
                val2 = tl.load(operation_dense + block_offset, mask=mask, other=0)
                # do the arithmetic operation
                if operation_function == "add":
                    jagged_val = tensor_elementwise_add(val1, val2)
                else:
                    jagged_val = tensor_elementwise_mul(val1, val2)

            # store the result
            tl.store(output_dense_ptr + block_offset, jagged_val, mask=mask)

            # update the block offset
            offset_col += thread_block_col_size
            block_offset += thread_block_col_size
        offset_row += thread_block_row_size


# Migrated from recsys-examples/examples/hstu/ops/triton_ops/triton_silu.py
# Changes:
#   - Imports updated to src.kernels.recsys_example.utils
#   - fast_dividef replaced with native division for triton-ascend compatibility


# Source: Q3TritonKernel/src/kernels/recsys_example/silu/silu.py:29
@triton.jit
def _silu_forward(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    x_size: tl.int32,
    x_block_size: tl.constexpr,
):
    x_offset = tl.program_id(0) * x_block_size
    mask = x_offset + tl.arange(0, x_block_size) < x_size
    output_block_ptr = output_ptr + x_offset
    input_block_ptr = input_ptr + x_offset
    cols = tl.arange(0, x_block_size)

    input = tl.load(input_block_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    output = input / (1.0 + tl.exp(-input))
    tl.store(output_block_ptr + cols, output.to(output_ptr.dtype.element_ty), mask=mask)


# Migrated from recsys-examples/examples/hstu/ops/triton_ops/triton_layer_norm.py
# Changes:
#   - Imports updated to src.kernels.recsys_example.utils
#   - torch.library.wrap_triton(...) calls replaced with direct kernel[(grid,)](...) calls
#   - torch.cuda.get_device_properties(...) made NPU-aware via _get_device_properties


# Source: Q3TritonKernel/src/kernels/recsys_example/layer_norm/layer_norm.py:37
@triton.jit
def _layer_norm_fwd(
    X,
    Y,
    Mean,
    Rstd,
    D,
    eps,
    stride_x,
    stride_y,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x, axis=0) / D
    else:
        mean = tl.load(Mean + row)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = tl.zeros([BLOCK_D], dtype=tl.float32)
        _var += x_mean * x_mean
        var = tl.sum(_var, axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
    else:
        rstd = tl.load(Rstd + row)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    # Write output
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Source: Q3TritonKernel/src/kernels/fla/modules/l2norm.py:42
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D: tl.constexpr,
    BD: tl.constexpr,
    T: tl.constexpr = 0,
    BR: tl.constexpr = 1,
):
    # Keep the original one-row calculation when batching is not selected.
    if BR == 1:
        i_t = tl.program_id(0)
        x += i_t * D
        y += i_t * D
        # Compute mean and variance
        cols = tl.arange(0, BD)
        mask = cols < D

        b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
        b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
        b_y = b_x * b_rstd
        tl.store(y + cols, b_y, mask=mask)
        tl.store(rstd + i_t, b_rstd)
    else:
        tl.static_assert(BR == 2 or BR == 4 or BR == 8)
        tl.static_assert(T > 0)
        rows = tl.program_id(0) * BR + tl.arange(0, BR)
        cols = tl.arange(0, BD)
        offsets = rows[:, None] * D + cols[None, :]
        mask = cols[None, :] < D
        if T % BR != 0:
            mask = mask & (rows[:, None] < T)

        b_x = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        # Reduce each row independently; preserve sqrt followed by division.
        b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, axis=1) + eps)
        b_y = b_x * b_rstd[:, None]
        tl.store(y + offsets, b_y, mask=mask)
        if T % BR == 0:
            tl.store(rstd + rows, b_rstd)
        else:
            tl.store(rstd + rows, b_rstd, mask=rows < T)


# Source: Q3TritonKernel/src/kernels/minimax_m3/topk_index_kernel.py:18
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


# Source: Q3TritonKernel/src/kernels/minimax_m3/topk_index_kernel.py:43
@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


# Source: Q3TritonKernel/src/kernels/minimax_m3/topk_index_partial_kernel.py:43
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _topk_index_partial_kernel(
    s_ptr, ts_partial_ptr, ti_partial_ptr, seq_lens, block_size: tl.constexpr,
    topk: tl.constexpr, chunk_blocks, decode_query_len, stride_s_h, stride_s_b,
    stride_s_k, stride_ts_c, stride_ts_h, stride_ts_b, stride_ts_t,
    stride_ti_c, stride_ti_h, stride_ti_b, stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    seq_len = tl.load(seq_lens + req_id)
    kv_len = tl.maximum(seq_len - decode_query_len + q_offset + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = s_ptr + pid_b * stride_s_b + pid_h * stride_s_h + (chunk_start + off_k) * stride_s_k
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        score = tl.load(s_ptrs, mask=mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = tl.where(mask, chunk_start + i + off_k + 1, 0), topk_idx
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), j, 2, n_dims)
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims)
            topk_score_new = last_topk_score * left_half_mask + topk_score * (1 - left_half_mask)
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (1 - left_half_mask)
            topk_score, topk_idx = _bitonic_merge(topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims)
        else:
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims)
    extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(extract[:, None] * tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]), axis=0)
    final_idx = tl.sum(extract[:, None] * tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]), axis=0)
    ts_ptrs = ts_partial_ptr + pid_chunk * stride_ts_c + pid_b * stride_ts_b + pid_h * stride_ts_h + off_t * stride_ts_t
    ti_ptrs = ti_partial_ptr + pid_chunk * stride_ti_c + pid_b * stride_ti_b + pid_h * stride_ti_h + off_t * stride_ti_t
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Code adapted from https://github.com/mayank31398/cute-kernels


# Source: Q3TritonKernel/src/kernels/fla/ops/utils/pack.py:29
@triton.jit
def packunpack_sequence_kernel(
    x,
    y,
    cu_seqlens,
    S,
    D,
    BD: tl.constexpr,
    PADDING_SIDE: tl.constexpr,
    PACK: tl.constexpr,
):
    i_d, i_s, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int64), tl.load(cu_seqlens + i_b + 1).to(tl.int64)

    T = eos - bos
    if PADDING_SIDE == 'left':
        NP = S - T
        if i_s < NP:
            return
        i_t = bos + (i_s - NP)
    else:
        if i_s >= T:
            return
        i_t = bos + i_s

    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    if PACK:
        b_x = tl.load(x + (i_b * S + i_s) * D + o_d, mask=mask)
        tl.store(y + i_t * D + o_d, b_x, mask=mask)
    else:
        b_x = tl.load(x + i_t * D + o_d, mask=mask)
        tl.store(y + (i_b * S + i_s) * D + o_d, b_x, mask=mask)


# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Source: Q3TritonKernel/src/kernels/fla/ops/utils/pooling.py:30
@triton.jit(do_not_specialize=['T'])
def mean_pooling_fwd_kernel(
    x,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_x = (o_t[:, None] < T) & (o_d[None, :] < D)
    p_x = x + (bos * H + i_h) * D + o_t[:, None] * (H*D) + o_d[None, :]
    p_o = o + (i_tg * H + i_h) * D + o_d
    # [BT, BD]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    # [BD]
    b_o = tl.sum(b_x, axis=0) / min(BT, T - i_t * BT)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=o_d < D)
