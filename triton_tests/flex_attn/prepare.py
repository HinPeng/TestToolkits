import torch
import torch_npu
import torch_npu._inductor
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

DTYPE = torch.bfloat16
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
SLIDING_WINDOW = 1023
GLOBAL_WINDOW = 4

DATA_LENGTH = [[2000, 22000, 2000], [2000, 22000, 2000]]
DATA_INPUT_TYPE = [["text", "image_gen", "text"], ["text", "image_gen", "text"]]
FULL_MASK_MODALITIES = ["image_gen", "image_vae"]

SEED = 0
CORRECTNESS_RTOL = 2e-2
BLOCK_SIZE = 64
RETURN_GRID = torch.tensor((5200000), dtype=DTYPE, device='npu')

def _grad_snapshot_cpu(problem):
    return {
        name: problem[name].grad.detach().float().cpu()
        for name in ('q', 'k', 'v')
    }

def _all_correct(metrics):
    return all(v < CORRECTNESS_RTOL for v in metrics.values())

def _build_modality_indicators(device):
    indicator = []
    iidx = 1
    for sample_types, sample_lens in zip(DATA_INPUT_TYPE, DATA_LENGTH):
        for t, l in zip(sample_types, sample_lens):
            if t in FULL_MASK_MODALITIES:
                indicator.append(torch.full((l,), iidx, dtype=torch.long))
                iidx += 1
            else:
                indicator.append(torch.full((l,), -1, dtype=torch.long))
    return torch.cat(indicator).to(device)

def build_program():
    device = torch.device("npu")
    torch.manual_seed(SEED)

    sample_lens = [sum(s) for s in DATA_LENGTH]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_s = int(cu_seqlens[-1].item())
    print("total_s:", total_s)
    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), dtype=torch.int32, device=device),
        torch.tensor(sample_lens, device=device)
    )
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).to(torch.long)
    modality = _build_modality_indicators(device)

    q = torch.randn(1, total_s, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    k = torch.randn(1, total_s, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v = torch.randn(1, total_s, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

    return {
        "q": q, "k": k, "v": v,
        "segment_ids": segment_ids.long(),
        "modality": modality,
        "doc_start": doc_start,
        "cu_seqlens": cu_seqlens,
        "total_s": total_s,
        "sliding_window": SLIDING_WINDOW,
        "global_window": GLOBAL_WINDOW,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
    }

def _full_mask_mod(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        samedoc_causal = same_doc & causal
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return samedoc_causal | same_img

    return mask_mod

def _sparse_mask_mod(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= W)
        glob = causal & ((kv_idx >= doc_start[q_idx]) & (kv_idx < doc_start[q_idx] + G))
        sparse = same_doc & (window | glob)
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return sparse | same_img

    return mask_mod

_flex_compiled = None
def _flex_attention(problem, block_mask):
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    q = problem["q"].transpose(1, 2).contiguous()
    k = problem["k"].transpose(1, 2).contiguous()
    v = problem["v"].transpose(1, 2).contiguous()
    print(f'q.shape: {q.shape}')
    print(f'k.shape: {k.shape}')
    print(f'v.shape: {v.shape}')

    out = _flex_compiled(q, k, v, block_mask=block_mask, enable_gqa=True)
    return out.transpose(1, 2).unsqueeze(0)

def reference_full(problem):
    T = problem["total_s"]
    bm = create_block_mask(_full_mask_mod(problem), B=None, H=None, Q_LEN=T, KV_LEN=T,
                           BLOCK_SIZE=BLOCK_SIZE, _compile=False)
    return _flex_attention(problem, bm)

def reference_sparse(problem):
    T = problem["total_s"]
    bm = create_block_mask(_sparse_mask_mod(problem), B=None, H=None, Q_LEN=T, KV_LEN=T,
                           BLOCK_SIZE=BLOCK_SIZE, _compile=False)
    return _flex_attention(problem, bm)

def _loss(out):
    return out.mean()

def _enable_grad(problem):
    for name in ('q', 'k', 'v'):
        t = problem[name]
        t.requires_grad_(True)
        t.grad = None

def _disable_grad(problem):
    for name in ('q', 'k', 'v'):
        t = problem[name]
        t.grad = None
        t.requires_grad_(False)

def _zero_grad(problem):
    for name in ('q', 'k', 'v'):
        t = problem[name]
        if t.grad is not None:
            t.grad.zero_()

def _max_abs_diff(actual, ref):
    return (actual - ref).abs().max().item()

def _collect_metrics(fwd_out, ref_out, grads, ref_grads):
    return {
        "fwd_max_diff": _max_abs_diff(fwd_out, ref_out),
        "dq_max_diff": _max_abs_diff(grads['q'], ref_grads['q']),
        "dk_max_diff": _max_abs_diff(grads['k'], ref_grads['k']),
        "dv_max_diff": _max_abs_diff(grads['v'], ref_grads['v'])
    }


def _resolve_backend():
    return 'npu', torch.device('npu')

def _get_device_module():
    return torch.npu

def _empty_cache():
    torch.npu.empty_cache()

_ITERS = 3
_MB = 1024 * 1024

def _reset_peak_memory_status():
    device_mod = _get_device_module()
    if hasattr(device_mod, "reset_peak_memory_stats"):
        device_mod.reset_peak_memory_stats()

def _sync():
    device_mod = _get_device_module()
    if hasattr(device_mod, "synchronize"):
        device_mod.synchronize()

def _make_events():
    device_mod = _get_device_module()
    if hasattr(device_mod, "Event"):
        return device_mod.Event(enable_timing=True), device_mod.Event(enable_timing=True)
    return None, None

import time
def _elapsed_ms(start_event, end_event, start_time=None, end_time=None):
    if start_event is not None and end_event is not None:
        return start_event.elapsed_time(end_event)
    if start_time is None:
        start_time = time.perf_counter()
    if end_time is None:
        end_time = time.perf_counter()
    return (end_time - start_time) * 1000

def _max_memory_allocated():
    device_mod = _get_device_module()
    if hasattr(device_mod, "max_memory_allocated"):
        return device_mod.max_memory_allocated()
    return 0

def _time_and_peak(fn, iters):
    _empty_cache()
    _reset_peak_memory_status()
    _sync()
    start, end = _make_events()
    start_time = time.perf_counter() if start is None else None
    if start is not None:
        start.record()
    
    with torch.no_grad():
        for _ in range(iters):
            out = fn()
    end_time = time.perf_counter() if end is None else None
    if end is not None:
        end.record()
    _sync()
    avg_ms = _elapsed_ms(start, end, start_time, end_time) / iters
    peak_gb = _max_memory_allocated() / _MB
    return avg_ms, peak_gb, out

def _fwd_bwd_measure(fn, problem, iters, capture_grad=False):
    s, e = _make_events()

    _empty_cache()
    _reset_peak_memory_status()
    _sync()
    fwd_start_time = time.perf_counter() if s is None else None
    if s is not None:
        s.record()
    for _ in range(iters):
        out = fn()
        del out
    
    fwd_end_time = time.perf_counter() if e is None else None
    if e is not None:
        e.record()
    _sync()

    fwd_ms = _elapsed_ms(s, e, fwd_start_time, fwd_end_time) / iters
    fwd_peak_mb = _max_memory_allocated() / _MB

    out = fn()
    fwd_out_cpu = out.detach().float().cpu()

    _reset_peak_memory_status()
    _sync()
    bwd_start_time = time.perf_counter() if s is None else None
    if s is not None:
        s.record()
    
    grad_cpu =None
    if capture_grad:
        _loss(out).backward(RETURN_GRID)
        grad_cpu = _grad_snapshot_cpu(problem)
    else:
        _loss(out).backward()
    bwd_end_time = time.perf_counter() if e is None else None
    if e is not None:
        e.record()
    _sync()

    bwd_ms = _elapsed_ms(s, e, bwd_start_time, bwd_end_time)
    bwd_peak_mb = _max_memory_allocated() / _MB
    _zero_grad(problem)
    del out
    _empty_cache()

    for _ in range(iters - 1):
        o = fn()
        iter_start_time = time.perf_counter() if s is None else None
        if s is not None:
            s.record()
        _loss(o).backward()
        iter_end_time = time.perf_counter() if e is None else None
        if e is not None:
            e.record()
        _sync()

        bwd_ms += _elapsed_ms(s, e, iter_start_time, iter_end_time)
        _zero_grad(problem)
        del o
    bwd_ms = bwd_ms / iters
    _empty_cache()

    if capture_grad:
        return fwd_ms, fwd_peak_mb, bwd_ms, bwd_peak_mb, fwd_out_cpu, grad_cpu
    return fwd_ms, fwd_peak_mb, bwd_ms, bwd_peak_mb, fwd_out_cpu

def _run_scenario(build_fn, apply_fn, problem, ref, ref_grads):
    T = problem['total_s']
    _enable_grad(problem)

    m = build_fn(problem)
    o = apply_fn(problem, m)

    _loss(o).backward()
    _zero_grad(problem)
    del m, o
    _empty_cache()

    build_ms, build_peak_mb, mask = _time_and_peak(lambda: build_fn(problem), iters=1)

    fwd_ms, fwd_peak_mb, bwd_ms, bwd_peak_mb, fwd_out, grads = _fwd_bwd_measure(lambda: apply_fn(problem, mask), problem, iters=_ITERS, capture_grad=True)

    metrics =_collect_metrics(fwd_out, ref, grads, ref_grads)
    _disable_grad(problem)
    del mask
    _empty_cache()

    total_peak_mb = max(fwd_peak_mb, bwd_peak_mb, build_peak_mb)
    return {
        "build_ms": build_ms,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "total_ms": build_ms + fwd_ms + bwd_ms,
        "build_gb": build_peak_mb / 1024,
        "fwd_gb": fwd_peak_mb / 1024,
        "bwd_gb": bwd_peak_mb / 1024,
        "total_gb": total_peak_mb / 1024,
        **metrics,
        "correct": _all_correct(metrics)
    }

def _measure_flex(problem, block_mask, iters):
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    _enable_grad(problem)

    q = problem['q'].transpose(1, 2)
    k = problem['k'].transpose(1, 2)
    v = problem['v'].transpose(1, 2)

    def fn():
        return _flex_compiled(q, k, v, block_mask=block_mask, enable_gqa=True)

    o = fn()
    _loss(o).backward()
    _zero_grad(problem)
    del o
    _empty_cache()

    fwd_ms, fwd_peak_mb, bwd_ms, bwd_peak_mb, fwd_out = _fwd_bwd_measure(fn, problem, iters=_ITERS)
    _disable_grad(problem)
    _empty_cache()
    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "fwd_gb": fwd_peak_mb / 1024,
        "bwd_gb": bwd_peak_mb / 1024,
    }
import os
Q_BUILD_CHUNK = int(os.environ.get("Q_BUILD_CHUNK", "512"))

def create_dense_mask_padding(
    document_ids,
    modality_indicators,
    feature_indicators = None,
    sliding_window_size = None,
    cu_seqlens = None,
    global_window = None,
    text_agnostic = False,
    vis = False,
    **kwargs
):
    slen = document_ids.size(-1)
    device = document_ids.device

    if feature_indicators is not None and (feature_indicators > 0).all():
        feature_indicators = None
    
    sample_indicators = None
    if cu_seqlens is not None:
        sample_indicators = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff())
    
    mask = torch.empty((slen, slen), dtype=torch.bool, device=device)

    kv_idx = torch.arange(slen, device=device, dtype=torch.int32)[None, :]
    doc_kv = document_ids[None, :]
    mod_kv = modality_indicators[None, :]
    fut_kv = feature_indicators[None, :] if feature_indicators is not None else None
    for qs in range(0, slen, Q_BUILD_CHUNK):
        qe = min(qs + Q_BUILD_CHUNK, slen)
        q_idx = torch.arange(qs, qe, device=device, dtype=torch.int32)[:, None]
        doc_q = document_ids[qs: qe][:, None]
        mod_q = modality_indicators[qs: qe][:, None]
        casusal = q_idx >= kv_idx
        same_doc = doc_q == doc_kv

        if sliding_window_size is not None:
            sliding = casusal & ((q_idx - kv_idx) <= sliding_window_size)
            if global_window is not None and sample_indicators is not None:
                doc_start = sample_indicators[qs: qe][:, None]
                glob = casusal & (kv_idx >= doc_start) & (kv_idx < (doc_start + global_window))
                window = sliding | glob
            else:
                window = sliding
            samedoc_causal = same_doc & window
        else:
            samedoc_causal = same_doc & casusal
        
        same_img = (mod_q > 0) & (mod_q == mod_kv)
        m = samedoc_causal | same_img
        
        if fut_kv is not None:
            future_ok = (fut_kv < 0) | ((fut_kv > 0) & (q_idx < fut_kv))
            m &= future_ok
        
        mask[qs: qe] = m
    return mask[None, None]


def flex_attention_triton(problem, block_mask, mask_type="full"):
    q = problem['q'].transpose(1, 2).contiguous()
    k = problem['k'].transpose(1, 2).contiguous()
    v = problem['v'].transpose(1, 2).contiguous()

    document_ids = problem["segment_ids"]
    modality_indicators = problem["modality"]
    doc_start = problem["doc_start"]
    W = problem["sliding_window"]
    G = problem["global_window"]
    block_mask.segment_ids = document_ids.contiguous()
    block_mask.modality_indicators = modality_indicators.contiguous()

    if mask_type == "sparse":
        block_mask.doc_start = doc_start.to(torch.int64).contiguous()
        if W is not None:
            block_mask.sliding_window = W
        if G is not None:
            block_mask.global_window = G
    if mask_type == "full":
        mask = create_dense_mask_padding(document_ids, modality_indicators)
    else:
        from train import build_dense_mask_sparse
        mask = build_dense_mask_sparse(problem)[None, None]
    block_mask.dense_mask = mask

    from flex_attention_triton import flex_attention as flex_attention_dev
    out = flex_attention_dev(q, k, v, block_mask=block_mask, mask_type=mask_type)

    return out.transpose(1, 2).squeeze(0)


def _measure_triton(problem, block_mask, ref_out, mask_type):
    _enable_grad(problem)

    def fn():
        return flex_attention_triton(problem, block_mask, mask_type=mask_type)
    
    o = fn()
    _loss(o).backward()
    _zero_grad(problem)
    del o
    _empty_cache()

    fwd_ms, fwd_peak_mb, bwd_ms, bwd_peak_mb, fwd_out = _fwd_bwd_measure(fn, problem, iters=_ITERS) 

    _disable_grad(problem)
    _empty_cache()
    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "fwd_gb": fwd_peak_mb / 1024,
        "bwd_gb": bwd_peak_mb / 1024,
    }
    
enable_compile = False

def evaluate(build_full, apply_full, build_sparse, apply_sparse):
    problem = build_program()
    T = problem["total_s"]

    full_bm = create_block_mask(_full_mask_mod(problem), B=None, H=None, Q_LEN=T, KV_LEN=T, BLOCK_SIZE=BLOCK_SIZE, _compile=False, device='npu')
    sparse_bm = create_block_mask(_sparse_mask_mod(problem), B=None, H=None, Q_LEN=T, KV_LEN=T, BLOCK_SIZE=BLOCK_SIZE, _compile=False, device='npu')
    if enable_compile:
        with torch.no_grad():
            full_ref = _flex_attention(problem, full_bm).float().cpu()
            sparse_ref = _flex_attention(problem, sparse_bm).float().cpu()
    else:
        _enable_grad(problem)
        full_ref_dev = flex_attention_triton(problem, full_bm, mask_type="full")
        full_ref = full_ref_dev.detach().float().cpu()
        _loss(full_ref_dev).backward(RETURN_GRID)
        full_ref_grads = _grad_snapshot_cpu(problem)
        _zero_grad(problem)

        sparse_ref_dev = flex_attention_triton(problem, sparse_bm, mask_type="sparse")
        sparse_ref = sparse_ref_dev.detach().float().cpu()
        _loss(sparse_ref_dev).backward(RETURN_GRID)
        sparse_ref_grads = _grad_snapshot_cpu(problem)
        _zero_grad(problem)

        del full_ref_dev, sparse_ref_dev
        _disable_grad(problem)
    del full_bm, sparse_bm
    _empty_cache()


    full = _run_scenario(build_full, apply_full, problem, full_ref, full_ref_grads)
    sparse = _run_scenario(build_sparse, apply_sparse, problem, sparse_ref, sparse_ref_grads)

    full_bm = create_block_mask(_full_mask_mod(problem), B=None, H=None, Q_LEN=T, KV_LEN=T, BLOCK_SIZE=BLOCK_SIZE, _compile=False, device='npu')
    sparse_bm = create_block_mask(_sparse_mask_mod(problem), B=None, H=None, Q_LEN=T, KV_LEN=T, BLOCK_SIZE=BLOCK_SIZE, _compile=False, device='npu')

    if enable_compile:
        flex_full = _measure_flex(problem, full_bm, iters=_ITERS)
        flex_sparse = _measure_flex(problem, sparse_bm, iters=_ITERS)
    else:
        flex_full = _measure_triton(problem, full_bm, full_ref, mask_type="full")
        flex_sparse = _measure_triton(problem, sparse_bm, sparse_ref, mask_type="sparse")

    correct = full["correct"] and sparse["correct"]
    return {
        "full": full,
        "sparse": sparse,
        "flex_full": flex_full,
        "flex_sparse": flex_sparse,
        "peak_vram_gb": max(full["total_gb"], sparse["total_gb"]),
        "correct": correct,
        "seq_len": T,
        "tol": CORRECTNESS_RTOL,
    }

if __name__ == "__main__":
    p = build_program()
    rf = reference_full(p)
    rs = reference_sparse(p)
