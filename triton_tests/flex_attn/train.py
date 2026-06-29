import os

import torch
import torch_npu
import torch_npu._inductor
import torch.nn.functional as F

from prepare import evaluate

def _print_scenario(name, scenario):
    print(f'{name}_fwd_ms: {scenario["fwd_ms"]:.2f}')
    print(f'{name}_bwd_ms: {scenario["bwd_ms"]:.2f}')
    print(f'{name}_fwd_gb: {scenario["fwd_gb"]:.2f}')
    print(f'{name}_bwd_gb: {scenario["bwd_gb"]:.2f}')
    print(f'{name}_fwd_max_diff: {scenario["fwd_max_diff"]:.4f}')
    print(f'{name}_dq_max_diff: {scenario["dq_max_diff"]:.4f}')
    print(f'{name}_dk_max_diff: {scenario["dk_max_diff"]:.4f}')
    print(f'{name}_dv_max_diff: {scenario["dv_max_diff"]:.4f}')


def _print_flex(name, fx):
    print(f'{name}_fwd_ms: {fx["fwd_ms"]:.2f}')
    print(f'{name}_bwd_ms: {fx["bwd_ms"]:.2f}')
    print(f'{name}_fwd_gb: {fx["fwd_gb"]:.2f}')
    print(f'{name}_bwd_gb: {fx["bwd_gb"]:.2f}')

import torch.nn.functional as F
_COMPUTE_BLOCK_Q = 1024
def _sdpa_with_mask(problem, bool_mask):
    q = problem['q'].transpose(1, 2)
    k = problem['k'].transpose(1, 2)
    v = problem['v'].transpose(1, 2)
    T = q.shape[2]

    outs = []
    for i in range(0, T, _COMPUTE_BLOCK_Q):
        j = min(i + _COMPUTE_BLOCK_Q, T)
        row = bool_mask[i:j]
        col_any = row.any(dim=0)
        nz = col_any.nonzero(as_tuple=False)
        kmin = int(nz[0].item())
        kmax = int(nz[-1].item()) + 1
        out_blk = F.scaled_dot_product_attention(
            q[:, :, i:j], k[:, :, kmin:kmax], v[:, :, kmin:kmax],
            attn_mask=row[None, None, :, kmin:kmax],
            dropout_p=0.0,
            enable_gqa=True,
        )
        outs.append(out_blk)
    out = torch.cat(outs, dim=2)
    return out.transpose(1, 2).squeeze(0)

_BUILD_BLOCK_Q = 4096

def build_dense_mask_sparse(problem):
    seg = problem['segment_ids']
    modality = problem['modality']
    doc_start = problem['doc_start']
    w = problem['sliding_window']
    G = problem['global_window']
    T = problem['total_s']
    device = seg.device

    mask = torch.empty((T, T), dtype=torch.bool, device=device)
    kv_idx = torch.arange(T, device=device)[None, :]
    seg_kv = seg[None, :]
    mod_kv = modality[None, :]
    for i in range(0, T, _BUILD_BLOCK_Q):
        j = min(i + _BUILD_BLOCK_Q, T)
        q_idx = torch.arange(i, j, device=device)[:, None]
        seg_q = seg[i:j, None]
        mod_q = modality[i:j, None]
        ds_q = doc_start[i:j, None]

        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= w)
        glob = causal & (kv_idx >= ds_q) & (kv_idx < (ds_q + G))
        block = (seg_q == seg_kv) & (window | glob)
        block |= (mod_q > 0) & (mod_q == mod_kv)
        mask[i:j] = block
    
    return mask

def build_dense_mask_full(problem):
    seg = problem['segment_ids']
    modality = problem['modality']
    T = problem['total_s']
    device = seg.device

    mask = torch.empty((T, T), dtype=torch.bool, device=device)
    kv_idx = torch.arange(T, device=device)[None, :]
    seg_kv = seg[None, :]
    mod_kv = modality[None, :]

    for i in range(0, T, _BUILD_BLOCK_Q):
        j = min(i + _BUILD_BLOCK_Q, T)
        q_idx = torch.arange(i, j, device=device)[:, None]
        seg_q = seg[i:j, None]
        mod_q = modality[i:j, None]

        block = (seg_q == seg_kv) & (q_idx >= kv_idx)
        block |= (mod_q > 0) & (mod_q == mod_kv)
        mask[i:j] = block
    
    return mask

def build_mask_full(problem):
    return build_dense_mask_full(problem)

def apply_full(problem ,mask):
    return _sdpa_with_mask(problem, mask)

def build_mask_sparse(problem):
    return build_dense_mask_sparse(problem)

def apply_sparse(problem ,mask):
    return _sdpa_with_mask(problem, mask)

def main():
    res = evaluate(build_mask_full, apply_full, build_mask_sparse, apply_sparse)
    full, sparse = res['full'], res['sparse']
    flex_full, flex_sparse = res['flex_full'], res['flex_sparse']

    status = 'ok' if res['correct'] else 'incorrect'

    print('---')
    print(f'status: {status}')
    print(f'correct: {res["correct"]}')
    print(f'seq_len: {res["seq_len"]}')
    print(f'tol: {res["tol"]}')
    print(f'peak_vram_gb: {res["peak_vram_gb"]}')
    _print_scenario('full', full)
    _print_scenario('sparse', sparse)
    _print_flex('flex_full', flex_full)
    _print_flex('flex_sparse', flex_sparse)

if __name__ == "__main__":
    main() 
