import torch
import triton
import triton.language as tl

from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask
from typing import Optional
from torch import Tensor

TILE_BLOCK_SIZE = 64

def _get_num_aicore():
    device = torch.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(int(props.get('num_aicore', 1)), 1)

def _get_num_vector_core():
    device = torch.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(int(props.get('num_vectorcore', 1)), 1)

def _persistent_launch_config(num_tasks):
    num_tasks = max(int(num_tasks), 1)
    return (min(_get_num_aicore(), num_tasks),), num_tasks

@triton.jit
def load_dense_mask(
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    offs_m,
    offs_n,
    Q_LEN,
    KV_LEN,
):
    # stride_mask_m = stride_mask_m.to(tl.int64)
    # stride_mask_n = stride_mask_n.to(tl.int64)

    ptrs = DENSE_MASK + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
    return tl.load(ptrs, mask=valid, other=0)

@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_out_z", "stride_out_h",
        "stride_lse_z",
        "stride_kv_idx_m",
        "NUM_TASKS",
        "NUM_Q_BLOCKS",
        "Q_LEN",
        "KV_LEN",
    ]
)
def flex_attention_kernel(
    Q, K, V,
    KV_NUM_BLKS, KV_IDX, FULL_KV_NUM_BLKS, FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m, stride_mask_n,
    OUT, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_out_z, stride_out_h, stride_out_m, stride_out_k,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_kv_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        out_offset = off_z * stride_out_z + off_hq * stride_out_h
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        OUT_ptr = OUT + out_offset
        LSE_ptr = LSE + lse_offset

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0
        )

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n_load,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )

            k = tl.load(
                K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n_load[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0
            )
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0
            )

            k = tl.trans(k)

            qk = tl.dot(q, k, input_precision="ieee")
            qk *= SM_SCALE

            qk = tl.where(mask, qk, float("-inf"))
            qk = tl.where(offs_n_load[None, :] < KV_LEN, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

            alpha = tl.math.exp(m_i - m_ij_masked)
            p = tl.math.exp(qk - m_ij_masked[:, None])

            pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij
        
        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N
                
                offs_n_load = kv_start + tl.arange(0, BLOCK_N)
                k = tl.load(
                    K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n_load[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0
                )
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0
                )
                k = tl.trans(k)

                qk = tl.dot(q, k, input_precision="ieee")
                qk *= SM_SCALE
                qk = tl.where(offs_n_load[None, :] < KV_LEN, qk, float("-inf"))
                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                masked_out_rows = (m_ij == float("-inf"))
                m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

                alpha = tl.math.exp(m_i - m_ij_masked)
                p = tl.math.exp(qk - m_ij_masked[:, None])

                pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + pv
                m_i = m_ij

        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        out_mask = (offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM)
        tl.store(
            OUT_ptr + offs_m[:, None] * stride_out_m + offs_v[None, :] * stride_out_k,
            acc,
            mask=out_mask
        )

        lse = m_i + tl.math.log(l_i)
        tl.store(LSE_ptr + offs_m * stride_lse_m, lse, mask=offs_m < Q_LEN)

@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_lse_z",
        "stride_delta_z",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
        "stride_q_idx_m",
        "NUM_TASKS",
        "NUM_KV_BLOCKS",
        "Q_LEN",
        "KV_LEN",
    ]
)
def flex_attention_backward_dqdkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        offs_n = kv_start_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        k = tl.load(
            K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
            mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0
        )
        v = tl.load(
            V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
            mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
            other=0.0
        )

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = q_sparse_idx
        sparse_q_idx_offset = q_sparse_idx * stride_q_idx_m

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq = off_hkv * GQA_SHARED_HEADS + off_g
            off_hq = off_hq.to(tl.int64)
            q_offset = off_z * stride_qz + off_hq * stride_qh
            do_offset = off_z * stride_doz + off_hq * stride_doh
            lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
            delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

            Q_h = Q + q_offset
            DQ_h = DQ + q_offset
            DO_h = DO + do_offset
            LSE_h = LSE + lse_offset
            DELTA_h = DELTA + delta_offset

            q_indices = Q_IDX + sparse_q_idx_offset
            q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
            block_m_end = tl.minimum(
                q_num_blocks * sparse_q_multiple,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                propagate_nan=tl.PropagateNan.ALL
            )

            for start_m in range(0, block_m_end):
                blk_idx_in_list = start_m // sparse_q_multiple
                q_block = tl.load(q_indices + blk_idx_in_list)
                q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                offs_m = q_start + tl.arange(0, BLOCK_M)

                q = tl.load(
                    Q_h + offs_m[:, None] * stride_qm + offs_k[None, :]* stride_qk,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0
                )
                do = tl.load(
                    DO_h + offs_m[:, None] * stride_dom + offs_v[None, :]* stride_dok,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0
                )
                lse = tl.load(LSE_h + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
                delta = tl.load(DELTA_h + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0).to(V.dtype.element_ty)
                lse = tl.where(lse == float("-inf"), 0.0, lse)

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                mask = load_dense_mask(
                    DENSE_MASK,
                    stride_mask_m,
                    stride_mask_n,
                    offs_m,
                    offs_n,
                    Q_LEN=Q_LEN,
                    KV_LEN=KV_LEN
                )
                qk = tl.where(mask, qk, float("-inf"))
                qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None]).to(V.dtype.element_ty)
                dp = tl.dot(do, tl.trans(v), input_precision="ieee").to(V.dtype.element_ty)
                ds = (p * (dp - delta[:, None])).to(V.dtype.element_ty)
                ds = tl.where(mask, ds, 0.0)

                dq = tl.dot(ds.to(Q.dtype.element_ty), k, input_precision="ieee")
                tl.atomic_add(
                    DQ_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    dq,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM)
                )
                dk = tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, input_precision="ieee")
                tl.atomic_add(
                    DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                    dk,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM)
                )
                dv = tl.dot(tl.trans(p).to(V.dtype.element_ty), do, input_precision="ieee")
                tl.atomic_add(
                    DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                    dv,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM)
                )
            if HAS_FULL_BLOCKS:
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL
                )

                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)

                    q = tl.load(
                        Q_h + offs_m[:, None] * stride_qm + offs_k[None, :]* stride_qk,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                        other=0.0
                    )
                    do = tl.load(
                        DO_h + offs_m[:, None] * stride_dom + offs_v[None, :]* stride_dok,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                        other=0.0
                    )
                    lse = tl.load(LSE_h + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
                    delta = tl.load(DELTA_h + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0).to(V.dtype.element_ty)
                    lse = tl.where(lse == float("-inf"), 0.0, lse)

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE
                    qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])

                    dq = tl.dot(ds.to(Q.dtype.element_ty), k, input_precision="ieee")
                    tl.atomic_add(
                        DQ_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                        dq,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM)
                    )
                    dk = tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, input_precision="ieee")
                    tl.atomic_add(
                        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                        dk,
                        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM)
                    )
                    dv = tl.dot(tl.trans(p).to(V.dtype.element_ty), do, input_precision="ieee")
                    tl.atomic_add(
                        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                        dv,
                        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM)
                    )                    

class FlexAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        block_mask=None,
        score_mod=None,
        mask_type="full",
        doc_start=None,
        sliding_window=None,
        global_window=None,
    ):
        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape

        GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1

        SM_SCALE = 1.0 / (D ** 0.5)

        BLOCK_M = TILE_BLOCK_SIZE
        BLOCK_N = TILE_BLOCK_SIZE
        SPARSE_Q_BLOCK_SIZE = BLOCK_M
        SPARSE_KV_BLOCK_SIZE = BLOCK_N

        num_q_blocks = (M + SPARSE_Q_BLOCK_SIZE - 1) // SPARSE_Q_BLOCK_SIZE

        output = torch.empty_like(q)
        lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

        kv_num_blks = block_mask.kv_num_blocks
        kv_idx = block_mask.kv_indices
        full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
        full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

        q_num_blks = getattr(block_mask, "q_num_blocks", None)
        q_idx = getattr(block_mask, "q_indices", None)
        full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
        full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        kv_num_blks = kv_num_blks.contiguous()
        kv_idx = kv_idx.contiguous()
        full_kv_num_blks = full_kv_num_blks.contiguous()
        full_kv_idx = full_kv_idx.contiguous()

        q_num_blks = q_num_blks.contiguous()
        q_idx = q_idx.contiguous()
        full_q_num_blks = full_q_num_blks.contiguous()
        full_q_idx = full_q_idx.contiguous()

        dense_mask = getattr(block_mask, "dense_mask", None)
        num_tasks = num_q_blocks * Z * Hq
        grid, num_tasks = _persistent_launch_config(num_tasks)

        kv_num_blks = kv_num_blks.to(torch.int32)
        kv_idx = kv_idx.to(torch.int32)
        full_kv_num_blks = full_kv_num_blks.to(torch.int32)
        full_kv_idx = full_kv_idx.to(torch.int32)
        q_num_blks = q_num_blks.to(torch.int32)
        q_idx = q_idx.to(torch.int32)
        full_q_num_blks = full_q_num_blks.to(torch.int32)
        full_q_idx = full_q_idx.to(torch.int32)

        flex_attention_kernel[grid](q, k, v,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            dense_mask, dense_mask.stride(2), dense_mask.stride(3),
            output, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            kv_idx.stride(2),
            SM_SCALE=SM_SCALE,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_TASKS=num_tasks,
            NUM_Q_BLOCKS=num_q_blocks,
            Q_HEAD=Hq,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=True
            )
        
        ctx.save_for_backward(
            q, k, v, output, lse,
            dense_mask,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx,
        )
        ctx.mask_type = mask_type
        ctx.sliding_window = sliding_window
        ctx.global_window = global_window
        ctx.gqa_shared_heads = GQA_SHARED_HEADS
        ctx.sm_scale = SM_SCALE
        ctx.sparse_q_block_size = SPARSE_Q_BLOCK_SIZE
        ctx.sparse_kv_block_size = SPARSE_KV_BLOCK_SIZE
        ctx.has_full_blocks = True

        return output, lse
    
    @staticmethod
    def backward(ctx, grad_output, grad_lse=None):
        (
            q, k, v, output, lse,
            dense_mask,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx,
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape
        GQA_SHARED_HEADS = ctx.gqa_shared_heads
        
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

        BLOCK_M_DQ = TILE_BLOCK_SIZE
        BLOCK_N_DQ = TILE_BLOCK_SIZE
        BLOCK_M_DKDV = TILE_BLOCK_SIZE
        BLOCK_N_DKDV = TILE_BLOCK_SIZE

        num_q_blocks = triton.cdiv(M, BLOCK_M_DQ)
        num_kv_blocks = triton.cdiv(N, BLOCK_N_DKDV)
        grid_dkdv, num_tasks_dkdv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
        flex_attention_backward_dqdkdv_kernel[grid_dkdv](
            q, k, v, grad_output, lse, delta,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx,
            dense_mask, dense_mask.stride(2), dense_mask.stride(3),
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            q_idx.stride(2),
            SM_SCALE=ctx.sm_scale,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M_DKDV,
            BLOCK_N=BLOCK_N_DKDV,
            NUM_TASKS=num_tasks_dkdv,
            NUM_KV_BLOCKS=num_kv_blocks,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
            SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
            Q_LEN=M,
            KV_LEN=N,
            GQA_SHARED_HEADS=GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS=ctx.has_full_blocks,
            multibuffer=True,
            set_workspace_multibuffer=4
        )
        dq *= ctx.sm_scale
        dk *= ctx.sm_scale
    
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None

def flex_attention(
    q,
    k,
    v,
    block_mask=None,
    score_mod=None,
    return_lse=False,
    mask_type="full",
    doc_start=None,
    sliding_window=None,
    global_window=None,
):
    output, lse = FlexAttentionFunc.apply(
        q,
        k,
        v,
        block_mask,
        score_mod,
        mask_type,
        doc_start,
        sliding_window,
        global_window,
    )
    if return_lse:
        return output, lse
    return output

def _flex_attention_triton(q, k, v, mask, block_mask, dropout_rate=0.0, input_formate=None):
    if input_formate == 'head_first':
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    block_mask.dense_mask = mask
    out = flex_attention(q, k, v, block_mask=block_mask)
    return out.transpose(1, 2).squeeze(0)
