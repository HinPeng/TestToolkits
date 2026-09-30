# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Frozen JIT definitions from vllm-project/vllm-ascend, revision in sources.json.
# Only module imports / host wrappers are replaced. Function text is unchanged.
# See LICENSE and sources.json for provenance and definition hashes.

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cann import extension as _extension
except ImportError:
    _extension = tl


def _resolve(name):
    value = getattr(_extension, name, None)
    return value if value is not None else getattr(tl, name, None)


get_element = _resolve("get_element")
extract_slice = _resolve("extract_slice")
insert_slice = _resolve("insert_slice")


# Source: vllm_ascend/ops/triton/mul_add.py:7
@triton.jit
def muls_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    scale,  # Scale factor.
    n_elements,  # Size of the vector.
    n_blocks,  # Total number of blocks.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x * scale + y
        tl.store(output_ptr + offsets, output, mask=mask)


# Source: vllm_ascend/ops/triton/compute_slot_mapping.py:93
@triton.jit
def _compute_slot_mapping_request(
    start_idx,
    end_idx,
    req_idx,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    is_circular,
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(start_idx, end_idx, TILE_BLOCK_SIZE):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        if TOTAL_CP_WORLD_SIZE == 1:
            block_indices = tl.where(is_circular, 0, pos // block_size)
            slot_offsets = pos % block_size
        else:
            virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)

            block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size
            slot_offsets = local_block_offsets % block_size

        INT32_MAX = 2147483647
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        if TOTAL_CP_WORLD_SIZE == 1:
            relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        else:
            relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_circular & (pos < 0), PAD_ID, slot_ids)
        if TOTAL_CP_WORLD_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


# Source: vllm_ascend/ops/triton/compute_slot_mapping.py:155
@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_fused_groups_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    is_circular_ptr,
    HAS_CIRCULAR: tl.constexpr,
    PAD_ID: tl.constexpr,
    NUM_REQS: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,
    PARALLEL_TILES: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(0)
    programs_per_group: tl.constexpr = NUM_REQS * PARALLEL_TILES + 1
    group_idx = program_idx // programs_per_group
    group_program_idx = program_idx - group_idx * programs_per_group

    block_table_addr = tl.load(block_table_addrs_ptr + group_idx)
    slot_mapping_addr = tl.load(slot_mapping_addrs_ptr + group_idx)
    block_table_ptr = tl.cast(block_table_addr, tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.cast(slot_mapping_addr, tl.pointer_type(tl.int32))

    if group_program_idx == programs_per_group - 1:
        for i in range(num_tokens, max_num_tokens, TILE_BLOCK_SIZE):
            offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    req_idx = group_program_idx // PARALLEL_TILES
    tile_idx = group_program_idx - req_idx * PARALLEL_TILES
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    block_table_stride = tl.load(block_table_strides_ptr + group_idx)
    block_size = tl.load(block_sizes_ptr + group_idx)
    is_circular = tl.load(is_circular_ptr + group_idx) if HAS_CIRCULAR else False
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    for i in range(
        start_idx + tile_idx * TILE_BLOCK_SIZE,
        end_idx,
        TILE_BLOCK_SIZE * PARALLEL_TILES,
    ):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)
        block_indices = tl.where(is_circular, 0, pos // block_size)
        slot_offsets = pos % block_size

        INT32_MAX = 2147483647
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_circular & (pos < 0), PAD_ID, slot_ids)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


# Source: vllm_ascend/ops/triton/compute_slot_mapping.py:229
@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_fused_groups_adaptive_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_addrs_ptr,
    slot_mapping_addrs_ptr,
    block_table_strides_ptr,
    block_sizes_ptr,
    is_circular_ptr,
    HAS_CIRCULAR: tl.constexpr,
    PAD_ID: tl.constexpr,
    NUM_REQS: tl.constexpr,
    SMALL_TILE_BLOCK_SIZE: tl.constexpr,
    SMALL_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
    LARGE_BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(0)
    programs_per_group: tl.constexpr = NUM_REQS + 1
    group_idx = program_idx // programs_per_group
    group_program_idx = program_idx - group_idx * programs_per_group

    block_table_addr = tl.load(block_table_addrs_ptr + group_idx)
    slot_mapping_addr = tl.load(slot_mapping_addrs_ptr + group_idx)
    block_table_ptr = tl.cast(block_table_addr, tl.pointer_type(tl.int32))
    slot_mapping_ptr = tl.cast(slot_mapping_addr, tl.pointer_type(tl.int32))

    if group_program_idx == NUM_REQS:
        for i in range(num_tokens, max_num_tokens, 1024):
            offsets = i + tl.arange(0, 1024)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    req_idx = group_program_idx
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    block_table_stride = tl.load(block_table_strides_ptr + group_idx)
    block_size = tl.load(block_sizes_ptr + group_idx)
    is_circular = tl.load(is_circular_ptr + group_idx) if HAS_CIRCULAR else False
    request_tokens = end_idx - start_idx
    if request_tokens <= SMALL_TILE_BLOCK_SIZE:
        _compute_slot_mapping_request(
            start_idx,
            end_idx,
            req_idx,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            is_circular,
            1,
            1,
            1,
            0,
            1,
            PAD_ID,
            SMALL_TILE_BLOCK_SIZE,
            SMALL_BLOCK_TABLE_WINDOW_SIZE,
        )
    else:
        _compute_slot_mapping_request(
            start_idx,
            end_idx,
            req_idx,
            positions_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            slot_mapping_ptr,
            is_circular,
            1,
            1,
            1,
            0,
            1,
            PAD_ID,
            1024,
            LARGE_BLOCK_TABLE_WINDOW_SIZE,
        )


# Source: vllm_ascend/ops/triton/rms_norm.py:5
@triton.jit(
    do_not_specialize=[
        "total_batch",
    ]
)
def triton_rms_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    norm_output_ptr,
    variance_epsilon,
    total_batch,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(total_batch, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, total_batch)
    offset_d = tl.arange(0, DIM)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        mask_r = offset_row < total_batch
        mask_row = mask_r[:, None]
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_row)

        variance = tl.sum(x * x, axis=-1) / DIM
        output = x * tl.rsqrt(variance[:, None] + variance_epsilon)

        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_row)


# Source: vllm_ascend/ops/triton/dsa_cp.py:4
@triton.jit
def build_local_metadata_triton(
    query_start_loc_ptr,  # [num_reqs + 1], int32
    seq_lens_ptr,  # [num_reqs], int32
    local_query_start_loc_ptr,  # [max_num_seqs + 1], int32  (output, pre-zeroed)
    local_seq_lens_ptr,  # [max_num_seqs], int32      (output, pre-zeroed)
    local_start,
    local_end,
    num_reqs,
    start_pos_out_ptr,  # [max_num_seqs], int32      (output)
    BLOCK_NUM_REQS: tl.constexpr,
    COMPUTE_START_POS: tl.constexpr,
):
    """Fused NPU kernel for local token metadata computation

    reduce kernel launch overhead.
    """
    offsets = tl.arange(0, BLOCK_NUM_REQS)
    mask = offsets < num_reqs

    q_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    q_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)

    lqs = tl.maximum(tl.minimum(q_start, local_end), local_start)
    lqe = tl.maximum(tl.minimum(q_end, local_end), local_start)
    lql = lqe - lqs

    cum = tl.cumsum(lql, axis=0)

    tl.store(local_query_start_loc_ptr + 1 + offsets, cum, mask=mask)

    offset = q_end - lqe
    result = tl.where((lql > 0) & (seq_len > 0), tl.maximum(seq_len - offset, 0), 0)
    tl.store(local_seq_lens_ptr + offsets, result, mask=mask)

    if COMPUTE_START_POS:
        tl.store(start_pos_out_ptr + offsets, seq_len - (q_end - q_start), mask=mask)


# Source: vllm_ascend/ops/triton/bincount.py:31
@triton.jit(
    do_not_specialize=[
        "tokens_batch_stride",
        "batch_size",
        "seq_len",
        "total_blocks",
    ]
)
def token_bin_counts_and_mask_kernel(
    tokens_ptr,
    tokens_batch_stride,
    tokens_seq_stride,
    batch_size,
    seq_len,
    vocab_size,
    bin_counts_ptr,
    tp_rank,
    counts_batch_stride,
    counts_vocab_stride,
    total_blocks,
    SEQ_BLOCK: tl.constexpr,
):
    """Count token occurrences per batch row.

    1D grid with grid-stride loop: each program processes blocks at
    stride=num_programs to stay within the Triton-Ascend coreDim
    limit (65535) while distributing work evenly across cores.
    """
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)

    vocab_start_idx = tp_rank * vocab_size
    n_seq_blocks = tl.cdiv(seq_len, SEQ_BLOCK)

    for linear_block in tl.range(pid, total_blocks, num_progs):
        batch_idx = linear_block // n_seq_blocks
        seq_block_id = linear_block - batch_idx * n_seq_blocks
        seq_start = seq_block_id * SEQ_BLOCK

        batch_tokens_start = tokens_ptr + batch_idx * tokens_batch_stride
        batch_counts_start = bin_counts_ptr + batch_idx * counts_batch_stride

        pos_offsets = seq_start + tl.arange(0, SEQ_BLOCK)
        pos_mask = pos_offsets < seq_len
        token = tl.load(
            batch_tokens_start + pos_offsets * tokens_seq_stride,
            mask=pos_mask,
            other=vocab_size + vocab_start_idx,
        )

        local_token = token - vocab_start_idx
        token_in_range = pos_mask & (token >= vocab_start_idx) & (local_token < vocab_size)

        safe_local_token = tl.where(token_in_range, local_token, 0)
        count_ptr = batch_counts_start + safe_local_token * counts_vocab_stride
        tl.atomic_add(count_ptr, 1, mask=token_in_range)


# Source: vllm_ascend/ops/triton/reject_sample.py:171
@triton.jit(
    do_not_specialize=[
        "max_spec_len",
        "vec_len",
    ]
)
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size] or [num_tokens, selected_vocab_size] if ENABLE_REDUCE_SAMPLING
    target_indices_ptr,  # [num_tokens, selected_vocab_size] global vocab indices, only used if ENABLE_REDUCE_SAMPLING
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,  # vocab_size or selected_vocab_size if ENABLE_REDUCE_SAMPLING
    global_vocab_size,  # global vocab size for draft_probs indexing (only used if ENABLE_REDUCE_SAMPLING)
    vec_len,
    ori_target_probs_ptr,  # [num_tokens, ori_vocab_size] original probs for entropy
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,  # Whether using reduce sampling
    SYNTHETIC_MODE: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
    POSTERIOR_THRESHOLD: tl.constexpr = 0.95,
    POSTERIOR_ALPHA: tl.constexpr = 0.4,
    SUB_BLOCK: tl.constexpr = 4096,
    EPSILON: tl.constexpr = 1e-10,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offsets == 0 must
    # be masked out at the load.
    start_idxs = tl.load(cu_num_draft_tokens_ptr + offsets - 1, mask=not_greedy_mask & (offsets > 0), other=0)
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask, other=0)
    n_num_draft_tokens = end_idxs - start_idxs

    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

            for pos in range(num_draft_tokens):
                if not rejected:
                    if SYNTHETIC_MODE:
                        # Synthetic: accept draft token with prob
                        # conditional_rates[pos], bypassing target/draft prob
                        # comparison. Output may be incorrect - benchmarking only.
                        token_idx = start_idx + pos
                        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
                        uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                        rate = tl.load(synthetic_conditional_rates_ptr + pos)
                        accepted = (uniform_prob < rate) & (draft_token_id >= 0)
                        if accepted:
                            tl.store(
                                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                                draft_token_id,
                            )
                        else:
                            rejected = True
                            recovered = tl.load(recovered_token_ids_ptr + token_idx)
                            tl.store(
                                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                                recovered,
                            )
                    elif ENABLE_REDUCE_SAMPLING:
                        token_idx = start_idx + pos
                        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

                        if draft_token_id == -1:
                            rejected = True
                            token_id = tl.load(recovered_token_ids_ptr + token_idx)
                        else:
                            target_prob = 0.0
                            found = False

                            for v_offset in range(0, vocab_size, VOCAB_BLOCK_SIZE):
                                if not found:
                                    vocab_offsets = v_offset + tl.arange(0, VOCAB_BLOCK_SIZE)
                                    vocab_mask = vocab_offsets < vocab_size

                                    candidate_indices = tl.load(
                                        target_indices_ptr + token_idx * vocab_size + vocab_offsets,
                                        mask=vocab_mask,
                                        other=-1,
                                    )

                                    match_mask = candidate_indices == draft_token_id

                                    candidate_probs = tl.load(
                                        target_probs_ptr + token_idx * vocab_size + vocab_offsets,
                                        mask=vocab_mask,
                                        other=0.0,
                                    )

                                    current_match_prob = tl.sum(candidate_probs * match_mask, axis=0)
                                    if current_match_prob > 0.0:
                                        target_prob = current_match_prob
                                        found = True

                            if NO_DRAFT_PROBS:
                                draft_prob = 1
                            else:
                                draft_prob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + draft_token_id)

                            uniform_prob = tl.load(uniform_probs_ptr + token_idx)

                            # Acceptance condition
                            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                                # Accept
                                token_id = draft_token_id
                            else:
                                # Reject - use recovered token
                                rejected = True
                                token_id = tl.load(recovered_token_ids_ptr + token_idx)

                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
                    else:
                        token_idx = start_idx + pos
                        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
                        if draft_token_id == -1:
                            rejected = True
                            token_id = tl.load(recovered_token_ids_ptr + token_idx)
                        else:
                            target_prob = tl.load(target_probs_ptr + token_idx * global_vocab_size + draft_token_id)
                            if NO_DRAFT_PROBS:
                                draft_prob = 1
                            else:
                                draft_prob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + draft_token_id)
                            uniform_prob = tl.load(uniform_probs_ptr + token_idx)

                            if ENTROPY_VERIFY:
                                loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
                                entropy = 0.0
                                for loop_i in range(loop):
                                    vocab_start = loop_i * SUB_BLOCK
                                    vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                                    vocab_mask = vocab_offset < vocab_size
                                    if NO_ORI_TARGET_PROBS:
                                        probs = tl.load(
                                            target_probs_ptr + token_idx * vocab_size + vocab_offset,
                                            vocab_mask,
                                            other=0,
                                        )
                                    else:
                                        probs = tl.load(
                                            ori_target_probs_ptr + token_idx * vocab_size + vocab_offset,
                                            vocab_mask,
                                            other=0,
                                        )
                                    log_probs = tl.log(probs + EPSILON)
                                    entropy_contrib = -probs * log_probs
                                    entropy += tl.sum(entropy_contrib)

                                exp_neg_entropy = tl.exp(-entropy * POSTERIOR_ALPHA)
                                threshold_by_entropy = exp_neg_entropy
                                threshold = tl.minimum(threshold_by_entropy, POSTERIOR_THRESHOLD)
                                _uniform_prob = threshold * uniform_prob
                            else:
                                _uniform_prob = uniform_prob
                            # NOTE(woosuk): While the draft probability should never be 0,
                            # we check it to avoid NaNs. If it happens to be 0, we reject.
                            if draft_prob > 0 and target_prob / draft_prob >= _uniform_prob:
                                # Accept.
                                token_id = draft_token_id
                            else:
                                # Reject. Use recovered token.
                                rejected = True
                                token_id = tl.load(recovered_token_ids_ptr + token_idx)
                        tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            if not rejected:
                # If all tokens are accepted, append the bonus token.
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                    bonus_token_id,
                )


# Source: vllm_ascend/ops/triton/reject_sample.py:364
@triton.jit(do_not_specialize=["replace_from", "replace_to", "vec_len"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    vec_len,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offset = req_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vec_len

    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offset == 0 must
    # be masked out at the load. other=0 keeps num_tokens deterministic (0)
    # for out-of-range lanes, which the store mask below relies on.
    start_idx = tl.load(cu_num_tokens_ptr + offset - 1, mask=len_mask & (offset > 0), other=0)
    end_idx = tl.load(cu_num_tokens_ptr + offset, len_mask, other=0)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + offset, len_mask)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    for i in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_tokens, (i,))
        start_idx1 = get_element(start_idx, (i,))
        src_val1 = get_element(src_val, (i,))
        offset1 = tl.arange(0, MAX_NUM_TOKENS)
        tl.store(output_ptr + start_idx1 + offset1, src_val1, mask=offset1 < num_tokens1)


# Source: vllm_ascend/ops/triton/spec_decode/ngram.py:6
@triton.jit
def ngram_spec_decode_kernel(
    token_ids_ptr,
    num_tokens_ptr,
    sampled_ptr,
    discard_ptr,
    next_token_ids_ptr,
    draft_token_ids_ptr,
    num_valid_draft_ptr,
    raw_valid_count_ptr,
    max_seq_len: tl.constexpr,
    max_new_tokens: tl.constexpr,
    vocab_size: tl.constexpr,
    min_n: tl.constexpr,
    max_n: tl.constexpr,
    k: tl.constexpr,
    batch_size,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    BLOCK: tl.constexpr = 1024 if max_n <= 5 else (512 if max_n <= 10 else (256 if max_n <= 16 else 128))
    NUM_BLOCKS: tl.constexpr = (max_seq_len + BLOCK - 1) // BLOCK
    NO_MATCH_F: tl.constexpr = 1.0e9

    for batch_idx in range(pid, batch_size, num_cores):
        seq_len = tl.load(num_tokens_ptr + batch_idx)
        discard = tl.load(discard_ptr + batch_idx)
        row_off = batch_idx * max_seq_len

        # ── Filter phase ────────────────────────────────────────────────
        s_off = tl.arange(0, max_new_tokens)
        sampled_vals = tl.load(
            sampled_ptr + batch_idx * max_new_tokens + s_off,
            care_padding=False,
        )

        if discard != 0:
            filtered = tl.full([max_new_tokens], -1, tl.int32)
            valid_count = 0
        else:
            is_valid = (sampled_vals != -1) & (sampled_vals < vocab_size)
            filtered = tl.where(is_valid, sampled_vals, tl.full([max_new_tokens], -1, tl.int32))
            valid_count = tl.sum(tl.cast(is_valid, tl.int32))

        tl.store(raw_valid_count_ptr + batch_idx, valid_count)

        avail_space = max_seq_len - seq_len
        if avail_space < 0:
            avail_space = 0
        if valid_count > avail_space:
            valid_count = avail_space

        nt = seq_len + valid_count

        # next_token = filtered[valid_count - 1] (masked-sum extraction).
        # Stored to HBM immediately to keep its live register short.
        if valid_count > 0:
            sel = s_off == (valid_count - 1)
            next_token = tl.sum(tl.where(sel, filtered, 0), axis=0)
        else:
            backup_pos = seq_len - 1
            if backup_pos < 0:
                backup_pos = 0
            next_token = tl.load(token_ids_ptr + row_off + backup_pos)
        tl.store(next_token_ids_ptr + batch_idx, next_token)

        # Append the valid prefix to token_ids; the matching phase below searches
        # this row including the appended suffix.
        if valid_count > 0:
            c_mask = s_off < valid_count
            tl.store(
                token_ids_ptr + row_off + seq_len + s_off,
                filtered,
                mask=c_mask,
            )

        # ── Longest n-gram match + draft extraction phase ──────────────
        # Each block overlap-reads at most max_n tokens past its owned range to
        # evaluate every n-gram length purely in registers.
        #
        # For owned position i, L[i] = longest n in [min_n, max_n] such that
        #   token_ids[i .. i+n-1] == token_ids[nt-n .. nt-1]   (and i < nt-n).
        # best = globally longest L, tie-broken by earliest pos.
        #
        # For length n, D_n[i] = OR_{j<n} (token_ids[i+j] - suffix_n[j]) with
        # suffix_n[j] = token_ids[nt-n+j]. D_n[i] == 0 iff the window matches. The
        # inner j loop is unrolled (tl.static_range) into a single int32
        # accumulator `d` per n — fully vectorized, never scalar.
        best_pos = -1
        best_len = 0

        if valid_count > 0 and nt >= min_n:
            g_best_len = 0.0
            g_best_pos = NO_MATCH_F

            if NUM_BLOCKS > 1:
                s_tail = tl.arange(0, max_n)
                s_pos = nt - max_n + s_tail
                s_pos = tl.where(s_pos < 0, 0, s_pos)
                S_tail = tl.load(token_ids_ptr + row_off + s_pos)

            for bidx in tl.range(NUM_BLOCKS):
                base = bidx * BLOCK
                off = tl.arange(0, BLOCK)
                pos = base + off
                pos_f = tl.cast(pos, tl.float32)

                L = tl.full([BLOCK], 0.0, tl.float32)
                for n_val in tl.static_range(min_n, max_n + 1):
                    d = tl.full([BLOCK], 0, tl.int32)
                    for j in tl.static_range(0, n_val):
                        gp = pos + j
                        tok_j = tl.load(
                            token_ids_ptr + row_off + gp,
                            mask=gp < nt,
                            care_padding=False,
                        )
                        if NUM_BLOCKS > 1:
                            # Extract S_tail[max_n - n_val + j] via masked selection.
                            sel_mask = s_tail == (max_n - n_val + j)
                            suf = tl.sum(tl.where(sel_mask, S_tail, 0), axis=0)
                        else:
                            # Single-block path: a direct scalar load is cheaper than
                            # the masked-selection overhead.
                            sidx = nt - n_val + j
                            sidx = tl.where(sidx < 0, 0, sidx)
                            suf = tl.load(token_ids_ptr + row_off + sidx)
                        d = d | (tok_j - suf)
                    match_n = d == 0
                    valid_n = pos_f < (nt - n_val)
                    L = tl.where(match_n & valid_n, tl.cast(n_val, tl.float32), L)

                # Per-block best: longest L, tie-break earliest pos.
                block_best_len = tl.max(L, axis=0)
                eq_best = block_best_len == L
                has_match = block_best_len > 0.0
                cand_pos = tl.where(eq_best & has_match, pos_f, NO_MATCH_F)
                block_best_pos = tl.min(cand_pos, axis=0)

                # Reduce into the global best (longer wins; equal -> earlier pos).
                new_better = block_best_len > g_best_len
                same_earlier = (block_best_len == g_best_len) & (block_best_pos < g_best_pos)
                update = new_better | same_earlier
                g_best_len = tl.where(update, block_best_len, g_best_len)
                g_best_pos = tl.where(update, block_best_pos, g_best_pos)

            if g_best_len > 0.0:
                best_pos = tl.cast(g_best_pos, tl.int32)
                best_len = tl.cast(g_best_len, tl.int32)

        # Draft extraction (vectorized).
        draft_start = best_pos + best_len
        tokens_avail = nt - draft_start
        if tokens_avail < 0:
            tokens_avail = 0

        d_off = tl.arange(0, k)
        if best_pos >= 0:
            can_copy = d_off < tokens_avail
            draft_vals = tl.load(
                token_ids_ptr + row_off + draft_start + d_off,
                mask=can_copy,
                other=-1,
                care_padding=False,
            )
            tl.store(draft_token_ids_ptr + batch_idx * k + d_off, draft_vals)
            # Draft tokens copied from the sequence are real (>= 0) token ids, so
            # the valid count is simply min(k, tokens_avail) — no reload/reduce.
            valid_draft = tokens_avail
            if valid_draft > k:
                valid_draft = k
        else:
            tl.store(
                draft_token_ids_ptr + batch_idx * k + d_off,
                tl.full([k], -1, tl.int32),
            )
            valid_draft = 0

        tl.store(num_valid_draft_ptr + batch_idx, valid_draft)


# Source: vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py:25
@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    positions_gm_ptr,
    cos_sin_cache_gm_ptr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size

    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    pos_indices = input_batch_offset + tl.arange(0, batch_size_per_iter_per_vec)
    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
    sin_cos_range = tl.arange(0, ROPE_DIM)
    cos_sin_cache_offset = cos_sin_cache_gm_ptr + sin_cos_range

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        x = tl.load(
            positions_gm_ptr + pos_indices + pos_offset, mask=(pos_indices + pos_offset) < input_batch_offset_end
        )
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM)
        if BIAS:
            q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
            k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

        values_tmp3 = tl.zeros((batch_size_per_iter_per_vec, ROPE_DIM), dtype=tl.bfloat16)
        for i in tl.range(batch_size_per_iter_per_vec):
            pos = get_element(x, (i,))
            values_tmp3 = insert_slice(
                values_tmp3.reshape(batch_size_per_iter_per_vec, ROPE_DIM),
                tl.load(pos * ROPE_DIM + cos_sin_cache_offset[:, None]).reshape(1, ROPE_DIM),
                offsets=(i, 0),
                sizes=(1, ROPE_DIM),
                strides=(1, 1),
            )
        values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        cos = extract_slice(
            values_tmp3,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        sin = extract_slice(
            values_tmp3,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        normalized_values_tmp = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values + q_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

        # q rope
        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16)
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        q_output_idx = output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
        mask = (mmask[:, None]) & (output_q_nmask[None, :])
        if IS_PARTIAL_ROPE:
            normalized_values_tmp = insert_slice(
                normalized_values_tmp,
                values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                q_gm_ptr + q_output_idx,
                normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )
        else:
            tl.store(
                q_gm_ptr + q_output_idx,
                values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=mask,
            )

        # k rope
        normalized_values_tmp1 = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )

        if BIAS:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values + k_bias_values).to(tl.bfloat16)
        else:
            normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16)

        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        kv_output_idx = output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
        mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        if IS_PARTIAL_ROPE:
            normalized_values_tmp1 = insert_slice(
                normalized_values_tmp1,
                values_tmp2,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                k_gm_ptr + kv_output_idx,
                normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mask,
            )
        else:
            tl.store(
                k_gm_ptr + kv_output_idx,
                values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=mask,
            )

    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size

    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)
        out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
        mblk_idx += v_batch_size_per_iter_per_vec


# Source: vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py:26
@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    cos_sin_ptr: torch.Tensor,
    out_q_ptr: torch.Tensor,
    out_k_ptr: torch.Tensor,
    out_v_ptr: torch.Tensor,
    out_gate_ptr: torch.Tensor,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_t,
    mrope_section_h,
    mrope_section_w,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    gate_size: tl.constexpr,
):
    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    for index in range(loop_num):
        ## load ##
        # q
        in_q_offset = in_qkv_ptr + (block_offset + index) * (q_size + gate_size + 2 * kv_size)
        if gate_size > 0:
            in_q_gate_tensor = (
                tl.load(in_q_offset + tl.arange(0, q_size + gate_size))
                .to(tl.float32)
                .reshape(num_q_heads, head_size * 2)
            )
            in_q_tensor = extract_slice(
                in_q_gate_tensor,
                offsets=(0, 0),
                sizes=(num_q_heads, head_size),
                strides=(1, 1),
            )
            in_gate_tensor = extract_slice(
                in_q_gate_tensor,
                offsets=(0, head_size),
                sizes=(num_q_heads, head_size),
                strides=(1, 1),
            ).reshape(q_size)
        else:
            in_q_tensor = tl.load(in_q_offset + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)

        # k
        in_k_offset = in_q_offset + q_size + gate_size
        in_k_tensor = tl.load(in_k_offset + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)
        # v
        in_v_offset = in_k_offset + kv_size
        in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))

        # cos, sin
        cos_offsets = tl.arange(0, half_rope_dim)
        if is_interleaved:
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
        else:
            t_mask = cos_offsets < mrope_section_t
            h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
            w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
                cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
            )

        t_cos_offset = cos_sin_ptr + (block_offset + index) * rope_dim
        h_cos_offset = t_cos_offset + num_tokens * rope_dim
        w_cos_offset = h_cos_offset + num_tokens * rope_dim

        t_sin_offset = cos_sin_ptr + (block_offset + index) * rope_dim + half_rope_dim
        h_sin_offset = t_sin_offset + num_tokens * rope_dim
        w_sin_offset = h_sin_offset + num_tokens * rope_dim

        t_cos_tensor = tl.load(t_cos_offset + cos_offsets, mask=t_mask, other=0)
        h_cos_tensor = tl.load(h_cos_offset + cos_offsets, mask=h_mask, other=0)
        w_cos_tensor = tl.load(w_cos_offset + cos_offsets, mask=w_mask, other=0)
        t_sin_tensor = tl.load(t_sin_offset + cos_offsets, mask=t_mask, other=0)
        h_sin_tensor = tl.load(h_sin_offset + cos_offsets, mask=h_mask, other=0)
        w_sin_tensor = tl.load(w_sin_offset + cos_offsets, mask=w_mask, other=0)

        cos_tensor = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32).reshape(1, half_rope_dim)
        cos_tensor = tl.broadcast_to(cos_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        sin_tensor = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32).reshape(1, half_rope_dim)
        sin_tensor = tl.broadcast_to(sin_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        ## compute ##
        # q-rmsnorm
        squares = in_q_tensor * in_q_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
        q_normalized = in_q_tensor * reciprocal_std
        q_normalized = q_normalized * q_rmsnorm_weight
        if has_bias:
            q_normalized = q_normalized + q_bias

        # k-rmsnorm
        squares = in_k_tensor * in_k_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
        k_normalized = in_k_tensor * reciprocal_std
        k_normalized = k_normalized * k_rmsnorm_weight
        if has_bias:
            k_normalized = k_normalized + k_bias

        # q-mrope
        x1 = extract_slice(
            q_normalized,
            offsets=(0, 0),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        x2 = extract_slice(
            q_normalized,
            offsets=(0, half_rope_dim),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_x = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
        cat_x = insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_x = insert_slice(
            cat_x,
            x1,
            offsets=(0, half_rope_dim),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        if IS_PARTIAL_ROPE:
            orig_qk = extract_slice(
                q_normalized,
                offsets=(0, 0),
                sizes=(num_q_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            orig_qk = q_normalized
        roped_q = cat_x * sin_tensor + orig_qk * cos_tensor

        # k-mrope
        y1 = extract_slice(
            k_normalized,
            offsets=(0, 0),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        y2 = extract_slice(
            k_normalized,
            offsets=(0, half_rope_dim),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_y = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
        cat_y = insert_slice(
            cat_y,
            -y2,
            offsets=(0, 0),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_y = insert_slice(
            cat_y,
            y1,
            offsets=(0, half_rope_dim),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        if IS_PARTIAL_ROPE:
            orig_qk = extract_slice(
                k_normalized,
                offsets=(0, 0),
                sizes=(num_kv_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            orig_qk = k_normalized
        roped_k = cat_y * sin_tensor + orig_qk * cos_tensor

        if IS_PARTIAL_ROPE:
            q_normalized = insert_slice(
                q_normalized,
                roped_q,
                offsets=(0, 0),
                sizes=(num_q_heads, rope_dim),
                strides=(1, 1),
            )
            k_normalized = insert_slice(
                k_normalized,
                roped_k,
                offsets=(0, 0),
                sizes=(num_kv_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            q_normalized = roped_q
            k_normalized = roped_k

        ## store ##
        # out_q
        out_q_offset = out_q_ptr + (block_offset + index) * q_size
        out_q_indices = tl.arange(0, q_size)
        tl.store(out_q_offset + out_q_indices, q_normalized.reshape(q_size))

        # out_k
        out_k_offset = out_k_ptr + (block_offset + index) * kv_size
        out_k_indices = tl.arange(0, kv_size)
        tl.store(out_k_offset + out_k_indices, k_normalized.reshape(kv_size))

        # out_v
        out_v_offset = out_v_ptr + (block_offset + index) * kv_size
        tl.store(out_v_offset + tl.arange(0, kv_size), in_v_tensor)

        # out_gate
        if gate_size > 0:
            out_gate_offset = out_gate_ptr + (block_offset + index) * gate_size
            tl.store(out_gate_offset + tl.arange(0, gate_size), in_gate_tensor)


# Source: vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py:39
@triton.jit
def split_qkv_rmsnorm_rope_simt_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_sin_precomputed_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
):
    pid = tl.program_id(0)

    batch_per_prog = tl.cdiv(batch_size, num_vectorcore)
    start = pid * batch_per_prog
    end = tl.minimum(start + batch_per_prog, batch_size)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size

    for iter in tl.range(tl.cdiv(end - start, batch_size_per_iter_per_vec)):
        base_batch = start + iter * batch_size_per_iter_per_vec
        batch_indices = base_batch + tl.arange(0, batch_size_per_iter_per_vec)
        mmask = batch_indices < end

        qk_cols = tl.arange(0, q_hidden_size + kv_hidden_size)
        mask = mmask[:, None] & (qk_cols[None, :] < total_hidden_size)

        idx = batch_indices[:, None] * total_hidden_size + qk_cols[None, :]
        values_tmp1 = (
            tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM).to(tl.float32)
        )

        if BIAS:
            q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
            k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

        cos_sin_offset = base_batch * ROPE_DIM + tl.arange(0, batch_size_per_iter_per_vec * ROPE_DIM)
        cos_sin_value = tl.load(
            cos_sin_precomputed_ptr + cos_sin_offset, mask=cos_sin_offset < (end * ROPE_DIM)
        ).reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)

        cos = extract_slice(
            cos_sin_value, offsets=(0, 0, 0), sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM), strides=(1, 1, 1)
        )
        sin = extract_slice(
            cos_sin_value,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        normalized_values_tmp = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            normalized_values_tmp = normalized_values_tmp * q_weight_values + q_bias_values
        else:
            normalized_values_tmp = normalized_values_tmp * q_weight_values

        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.float32)
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        q_output_idx = output_q_nblk_idx[None, :] + batch_indices[:, None] * q_hidden_size
        out_mask = mmask[:, None] & output_q_nmask[None, :]
        if IS_PARTIAL_ROPE:
            normalized_values_tmp = insert_slice(
                normalized_values_tmp,
                values_tmp,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, q_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                q_gm_ptr + q_output_idx,
                normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
                mask=out_mask,
            )
        else:
            tl.store(
                q_gm_ptr + q_output_idx, values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size), mask=out_mask
            )

        normalized_values_tmp1 = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            normalized_values_tmp1 = normalized_values_tmp1 * k_weight_values + k_bias_values
        else:
            normalized_values_tmp1 = normalized_values_tmp1 * k_weight_values

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.float32)
        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        kv_output_idx = output_kv_nblk_idx[None, :] + batch_indices[:, None] * kv_hidden_size
        out_mask = mmask[:, None] & output_kv_nmask[None, :]
        if IS_PARTIAL_ROPE:
            normalized_values_tmp1 = insert_slice(
                normalized_values_tmp1,
                values_tmp2,
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM),
                strides=(1, 1, 1),
            )
            tl.store(
                k_gm_ptr + kv_output_idx,
                normalized_values_tmp1.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=out_mask,
            )
        else:
            tl.store(
                k_gm_ptr + kv_output_idx,
                values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
                mask=out_mask,
            )

    for iter in tl.range(tl.cdiv(end - start, v_batch_size_per_iter_per_vec)):
        base_batch = start + iter * v_batch_size_per_iter_per_vec
        batch_indices = base_batch + tl.arange(0, v_batch_size_per_iter_per_vec)
        mmask = batch_indices < end

        v_cols = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
        nmask = v_cols < total_hidden_size
        mask = mmask[:, None] & nmask[None, :]

        idx = batch_indices[:, None] * total_hidden_size + v_cols[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)

        out_nblk_idx = tl.arange(0, kv_hidden_size)
        out_nmask = out_nblk_idx < kv_hidden_size
        out_idx = batch_indices[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = mmask[:, None] & out_nmask[None, :]
        tl.store(v_gm_ptr + out_idx, values, mask=out_mask)


# Source: vllm_ascend/ops/triton/rope.py:169
@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    cos_sin_ptr,
    cos_sin_row_stride,
    pos_ptr,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    USE_COS_SIN: tl.constexpr,
):
    """
    This triton kernel applies rotary embedding on q and k.
    It supports rope_dim != head_dim scenario.
    It supports both neox style and non-neox style rope computation.
    q/k head dimensions are tiled with BLOCK_SIZE_HEAD to avoid UB overflow.

    Input tensor layout assumptions:

    q size: (num_tokens, num_q_heads, head_dim)
    q stride: (num_q_heads * head_dim, head_dim, 1)
    k size: (num_tokens, num_kv_heads, head_dim)
    k stride: (num_kv_heads * head_dim, head_dim, 1)
    cos/sin size: (num_tokens, rope_dim/2)
    cos/sin stride: (rope_dim/2, 1)

    Different compute pattern of IS_NEOX_STYLE:

    if IS_NEOX_STYLE:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if IS_NEOX_STYLE:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)
    """
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        q_row_start_ptr = q_ptr + row_idx * q_row_stride
        k_row_start_ptr = k_ptr + row_idx * k_row_stride

        # ####################################################################
        # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
        # m of this program instance
        # ####################################################################
        cos_offsets = tl.arange(0, pad_rope_dim // 2)
        sin_offsets = cos_offsets + (rope_dim // 2)
        cos_mask = cos_offsets < (rope_dim // 2)
        if USE_COS_SIN:
            pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
            cos_start_ptr = cos_sin_ptr + pos_idx * cos_sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(cos_start_ptr + sin_offsets, mask=cos_mask, other=0).to(tl.float32)
        else:
            cos_start_ptr = cos_ptr + row_idx * cos_row_stride
            sin_start_ptr = sin_ptr + row_idx * sin_row_stride
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
            sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        # ####################################################################
        # Tile over q heads in chunks of BLOCK_SIZE_HEAD
        # ####################################################################
        for q_head_base in tl.range(0, n_qh, BLOCK_SIZE_HEAD):
            q_tile_start_ptr = q_row_start_ptr + q_head_base * hd
            q_heads = tl.arange(0, BLOCK_SIZE_HEAD)
            if IS_NEOX_STYLE:
                first_half_q_offsets = q_heads[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
                first_q_mask = ((q_head_base + q_heads)[:, None] < n_qh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
                )
                q_tile_1 = tl.load(q_tile_start_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
                    sin_row.dtype
                )
                second_half_q_offsets = first_half_q_offsets + (rope_dim // 2)
                second_q_mask = first_q_mask
                q_tile_2 = tl.load(q_tile_start_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(
                    sin_row.dtype
                )
                new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
                tl.store(q_tile_start_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
                new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
                tl.store(q_tile_start_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
            else:
                pair_offsets = (
                    q_heads[:, None, None] * hd
                    + (2 * tl.arange(0, pad_rope_dim // 2)[None, :, None])
                    + tl.arange(0, 2)[None, None, :]
                )
                pair_mask = ((q_head_base + q_heads)[:, None, None] < n_qh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :, None] < (rope_dim // 2)
                )
                q_tile = tl.load(q_tile_start_ptr + pair_offsets, mask=pair_mask, other=0).to(sin_row.dtype)
                q_tile_1, q_tile_2 = tl.split(q_tile)
                new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
                new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
                q_tile_out = tl.join(new_q_tile_1, new_q_tile_2)
                tl.store(q_tile_start_ptr + pair_offsets, q_tile_out, mask=pair_mask)

        # ####################################################################
        # Tile over k heads in chunks of BLOCK_SIZE_HEAD
        # ####################################################################
        for k_head_base in tl.range(0, n_kh, BLOCK_SIZE_HEAD):
            k_tile_start_ptr = k_row_start_ptr + k_head_base * hd
            k_heads = tl.arange(0, BLOCK_SIZE_HEAD)
            if IS_NEOX_STYLE:
                first_half_k_offsets = k_heads[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
                first_k_mask = ((k_head_base + k_heads)[:, None] < n_kh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :] < (rope_dim // 2)
                )
                k_tile_1 = tl.load(k_tile_start_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
                    sin_row.dtype
                )
                second_half_k_offsets = first_half_k_offsets + (rope_dim // 2)
                second_k_mask = first_k_mask
                k_tile_2 = tl.load(k_tile_start_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(
                    sin_row.dtype
                )
                new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
                tl.store(k_tile_start_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
                new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
                tl.store(k_tile_start_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
            else:
                pair_offsets = (
                    k_heads[:, None, None] * hd
                    + (2 * tl.arange(0, pad_rope_dim // 2)[None, :, None])
                    + tl.arange(0, 2)[None, None, :]
                )
                pair_mask = ((k_head_base + k_heads)[:, None, None] < n_kh) & (
                    tl.arange(0, pad_rope_dim // 2)[None, :, None] < (rope_dim // 2)
                )
                k_tile = tl.load(k_tile_start_ptr + pair_offsets, mask=pair_mask, other=0).to(sin_row.dtype)
                k_tile_1, k_tile_2 = tl.split(k_tile)

                new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
                new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
                k_tile_out = tl.join(new_k_tile_1, new_k_tile_2)
                tl.store(k_tile_start_ptr + pair_offsets, k_tile_out, mask=pair_mask)


# Source: vllm_ascend/ops/triton/activation/swiglu_quant.py:7
@triton.jit
def _swiglu_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
):
    # calc real total_rows
    if GROUP_LIST_TYPE == 0:  # cusum
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS - 1).to(tl.int32)
    else:
        gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
        gl_mask = gl_offsets < NUM_EXPERTS
        group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
        total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        x1 = extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        x2 = extract_slice(cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,))
        out = x1 * tl.sigmoid(x1) * x2

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            # store scale
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = extract_slice(out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,))
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            # store out
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


# Source: vllm_ascend/ops/triton/activation/swiglustep.py:44
@triton.jit
def _swiglustep_kernel(
    x_ptr,
    out_ptr,
    M,  # total rows (runtime value, not constexpr)
    TOTAL_COLS: tl.constexpr,  # 2N
    HALF_COLS: tl.constexpr,  # N
    LIMIT: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # even split of rows across cores; tail core handles the remainder
    block_size = (M - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= M:
        return
    row_end = tl.minimum((pid + 1) * block_size, M)

    col_offsets = tl.arange(0, TOTAL_COLS)
    out_offsets = tl.arange(0, HALF_COLS)

    for row_idx in range(row_begin, row_end):
        # load one full row [2N], split gate/up via extract_slice, compute in fp32
        x_row = tl.load(x_ptr + row_idx * TOTAL_COLS + col_offsets)
        gate = extract_slice(x_row, offsets=(0,), sizes=(HALF_COLS,), strides=(1,)).to(tl.float32)
        up = extract_slice(x_row, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)).to(tl.float32)

        # silu(gate) then clamp upper bound only
        s = gate * tl.sigmoid(gate)
        s = tl.minimum(s, LIMIT)
        # up clamp both sides
        up = tl.minimum(up, LIMIT)
        up = tl.maximum(-LIMIT, up)
        out = s * up

        tl.store(
            out_ptr + row_idx * HALF_COLS + out_offsets,
            out.to(x_ptr.dtype.element_ty),
        )


# Source: vllm_ascend/ops/triton/penalty.py:30
@triton.jit(
    do_not_specialize=[
        "num_seqs",
    ]
)
def apply_all_penalties_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    output_bin_counts_ptr,
    repetition_penalties_ptr,
    frequency_penalties_ptr,
    presence_penalties_ptr,
    num_seqs,
    vocab_size,
    stride_logits_seq,
    stride_logits_vocab,
    stride_prompt_mask_seq,
    stride_prompt_mask_vocab,
    stride_output_mask_seq,
    stride_output_mask_vocab,
    stride_bin_counts_seq,
    stride_bin_counts_vocab,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply repetition, frequency, and presence penalties to logits in place."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    seqs_per_program = (num_seqs + num_programs - 1) // num_programs

    start_seq = pid * seqs_per_program
    end_seq = tl.minimum(start_seq + seqs_per_program, num_seqs)

    for seq_idx in range(start_seq, end_seq):
        repetition_penalty = tl.load(repetition_penalties_ptr + seq_idx)
        frequency_penalty = tl.load(frequency_penalties_ptr + seq_idx)
        presence_penalty = tl.load(presence_penalties_ptr + seq_idx)

        for vocab_start in range(0, vocab_size, BLOCK_SIZE):
            vocab_offsets = vocab_start + tl.arange(0, BLOCK_SIZE)
            mask = vocab_offsets < vocab_size

            logits_offset = seq_idx * stride_logits_seq + vocab_offsets * stride_logits_vocab
            prompt_mask_offset = seq_idx * stride_prompt_mask_seq + vocab_offsets * stride_prompt_mask_vocab
            output_mask_offset = seq_idx * stride_output_mask_seq + vocab_offsets * stride_output_mask_vocab
            counts_offset = seq_idx * stride_bin_counts_seq + vocab_offsets * stride_bin_counts_vocab

            logits = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
            prompt_mask_val = tl.load(prompt_mask_ptr + prompt_mask_offset, mask=mask, other=False)
            output_mask_val = tl.load(output_mask_ptr + output_mask_offset, mask=mask, other=False)
            output_bin_counts = tl.load(
                output_bin_counts_ptr + counts_offset,
                mask=mask,
                other=0,
            ).to(tl.float32)

            need_repetition_penalty = (prompt_mask_val | output_mask_val).to(tl.int1)
            penalty_factor = tl.where(need_repetition_penalty, repetition_penalty, 1.0)
            scaling = tl.where(
                (logits > 0.0).to(tl.int1),
                1.0 / penalty_factor,
                penalty_factor,
            )
            updated = logits * scaling

            updated -= frequency_penalty * output_bin_counts
            updated -= presence_penalty * output_mask_val.to(tl.float32)
            tl.store(logits_ptr + logits_offset, updated, mask=mask)


# Source: vllm_ascend/ops/triton/spec_decode/utils.py:21
@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)


# Source: vllm_ascend/ops/triton/reject_sample.py:35
@triton.jit(do_not_specialize=["vec_len"])
def rejection_greedy_sample_spec_len_1_triton(
    output_token_ids_ptr,  # [batch_size, 2]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,
    vec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    draft_token_id = tl.load(draft_token_ids_ptr + offset, mask)
    target_argmax_id = tl.load(target_argmax_ptr + offset, mask)
    bonus_token_id = tl.load(bonus_token_ids_ptr + offset, mask)

    if SYNTHETIC_MODE:
        # Synthetic: accept the draft token with prob conditional_rates[0],
        # regardless of target match. Accepted => emit draft token (pos 0) and
        # the bonus token (pos 1); rejected => emit target_argmax (pos 0).
        uniform_prob = tl.load(uniform_probs_ptr + offset, mask)
        # spec_len == 1 => only position 0.
        rate = tl.load(synthetic_conditional_rates_ptr + 0)
        accepted = (uniform_prob < rate) & (draft_token_id >= 0) & mask
        # Cast both arms to int32: draft_token_id is int32, target_argmax_id is
        # int64 (from argmax); tl.where requires matching dtypes.
        token_id = tl.where(accepted, draft_token_id.to(tl.int32), target_argmax_id.to(tl.int32))
        tl.store(output_token_ids_ptr + offset * 2, token_id, mask)
        accept_mask = accepted
    else:
        tl.store(output_token_ids_ptr + offset * 2, target_argmax_id, mask)
        accept_mask = (draft_token_id == target_argmax_id) & mask
    tl.store(output_token_ids_ptr + offset * 2 + 1, bonus_token_id, accept_mask)


# Source: vllm_ascend/ops/triton/reject_sample.py:74
@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
    max_spec_len,
    num_tokens1,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1, bonus_token_id)


# Source: vllm_ascend/ops/triton/reject_sample.py:86
@triton.jit(do_not_specialize=["vec_len", "max_spec_len"])
def rejection_greedy_sample_triton(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    vec_len,
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    if is_greedy_ptr is None:
        is_greedy_mask = mask
    else:
        is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
        is_greedy_mask = mask & (is_greedy != 0)

    # Mask the load itself: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), so lane offset == 0 must
    # be masked out at the load. other=0 keeps num_draft_tokens deterministic
    # (0) for masked lanes, which the per-position loop below relies on to
    # skip them.
    start_idx = tl.load(cu_num_draft_tokens_ptr + offset - 1, mask=is_greedy_mask & (offset > 0), other=0)
    end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask, other=0)
    num_draft_tokens = end_idx - start_idx

    for pos in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_draft_tokens, (pos,))
        rejected = False
        start_idx1 = get_element(start_idx, (pos,))
        is_greedy_mask1 = get_element(is_greedy_mask, (pos,))
        position = block_idx * BLOCK_SIZE + pos
        for i in range(num_tokens1):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx1 + i)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx1 + i)
                if SYNTHETIC_MODE:
                    # Synthetic: accept draft token i with prob
                    # conditional_rates[i], independent of target match. Store
                    # each arm separately (draft on accept, target_argmax on
                    # reject) so the int32 (draft) / int64 (target_argmax)
                    # dtype mismatch is handled by the store's implicit cast --
                    # no ternary, no explicit cast (matches the random kernel's
                    # synthetic branch).
                    uniform_prob = tl.load(uniform_probs_ptr + start_idx1 + i)
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = (uniform_prob < rate) & (draft_token_id >= 0)
                    if accepted:
                        tl.store(
                            output_token_ids_ptr + position * (max_spec_len + 1) + i,
                            draft_token_id,
                        )
                    else:
                        tl.store(
                            output_token_ids_ptr + position * (max_spec_len + 1) + i,
                            target_argmax_id,
                        )
                        rejected = True
                else:
                    tl.store(
                        output_token_ids_ptr + position * (max_spec_len + 1) + i,
                        target_argmax_id,
                    )
                    if draft_token_id != target_argmax_id:
                        # Reject.
                        rejected = True

        if not rejected and is_greedy_mask1:
            bonus_renew(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
                max_spec_len,
                num_tokens1,
            )


# Source: vllm_ascend/ops/triton/reject_sample.py:398
@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    target_indices_ptr,
    q_ptr,
    vocab_size,
    global_vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)

    # Compute token index. Clamp the previous-request index instead of
    # relying on tl.where: tl.where does not prevent reading before the
    # buffer start (both arms are always evaluated), and a 0-d masked load is
    # not reliably supported on triton-ascend. The clamped load address always
    # stays inside the buffer; tl.where merely discards the value for
    # req_idx == 0.
    prev_req_idx = tl.maximum(req_idx - 1, 0)
    start_idx = tl.where(req_idx == 0, 0, tl.load(cu_num_draft_tokens_ptr + prev_req_idx))
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if ENABLE_REDUCE_SAMPLING:
        C = vocab_size
        n_loop = tl.cdiv(C, VOCAB_BLOCK_SIZE)

        global_max_p = tl.full((), -float("inf"), tl.float32)
        global_recovered_id = tl.full((), -1, tl.int64)
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int64)

        for li in range(n_loop):
            c_start = li * VOCAB_BLOCK_SIZE
            offs = c_start + tl.arange(0, VOCAB_BLOCK_SIZE)
            mask = offs < C

            # Load target prob and global index
            tprob = tl.load(target_probs_ptr + token_idx * C + offs, mask=mask, other=0.0).to(tl.float32)

            gidx = tl.load(target_indices_ptr + token_idx * C + offs, mask=mask, other=0).to(tl.int64)

            if NO_DRAFT_PROBS:
                is_draft = (gidx == draft_token_id) & mask
                prob = tl.where(is_draft, 0.0, tprob)
            else:
                valid = (gidx >= 0) & (gidx < global_vocab_size) & mask
                dprob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + gidx, mask=valid, other=0.0).to(
                    tl.float32
                )
                prob = tl.maximum(tprob - dprob, 0.0)

            qv = tl.load(q_ptr + req_idx * C + offs, mask=mask, other=1.0).to(tl.float32)

            bad_q = (qv <= 0) | (qv != qv) | (qv == float("inf")) | (qv == -float("inf"))
            score = tl.where(bad_q, float("-inf"), prob / qv)
            score = tl.where(mask, score, float("-inf"))

            block_best_score = tl.max(score, axis=0)
            block_best_idx = tl.argmax(score, axis=0).to(tl.int64)
            block_best_global_id = tl.load(target_indices_ptr + token_idx * C + (c_start + block_best_idx)).to(tl.int64)

            better = block_best_score > global_max_p
            global_max_p = tl.where(better, block_best_score, global_max_p)
            global_recovered_id = tl.where(better, block_best_global_id, global_recovered_id)

        tl.store(output_token_ids_ptr + token_idx, global_recovered_id)
    else:
        vocab_size = global_vocab_size
        loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        global_recovered_id = -1
        global_max_p = -1.0
        if NO_DRAFT_PROBS:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                    mask=vocab_offset < vocab_size,
                    other=0,
                )
                prob = tl.where(vocab_offset == draft_token_id, 0.0, prob)
                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=float("-inf")
                )
                new_p = prob / q
                recovered_id = tl.argmax(new_p, axis=-1)
                max_p = get_element(new_p, (recovered_id,))
                if max_p > global_max_p:
                    global_max_p = max_p
                    global_recovered_id = vocab_start + recovered_id
        else:
            for loop_i in range(loop):
                vocab_start = loop_i * SUB_BLOCK
                vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                    mask=vocab_offset < vocab_size,
                    other=0,
                )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset,
                    mask=vocab_offset < vocab_size,
                    other=0,
                )
                prob = tl.maximum(target_prob - draft_prob, 0)
                # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
                # `tl.argmax` will select the maximum value.

                q = tl.load(
                    q_ptr + req_idx * vocab_size + vocab_offset, mask=vocab_offset < vocab_size, other=float("-inf")
                )
                new_p = prob / q
                recovered_id = tl.argmax(new_p, axis=-1)
                max_p = get_element(new_p, (recovered_id,))
                if max_p > global_max_p:
                    global_max_p = max_p
                    global_recovered_id = vocab_start + recovered_id

        tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)
