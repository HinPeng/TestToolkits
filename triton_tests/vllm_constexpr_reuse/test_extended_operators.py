"""Additional vLLM / vLLM-MY operators from the required coverage list."""

import itertools

import pytest

import reference
from test_operators import guarded

pytestmark = pytest.mark.npu


def test_prepare_inputs_padded(case, kernels):
    torch = case.torch
    for step, (requests, block) in enumerate(((65, 32), (65, 64), (1, 64), (0, 32), (129, 32))):
        counts = [i % 4 for i in range(requests)]
        valid = [(i % (count + 1)) + 1 for i, count in enumerate(counts)]
        query = [0, *itertools.accumulate(count + 1 for count in counts)]
        indices, index_tail = guarded(case, (requests,), torch.int32)
        rejected, rejected_tail = guarded(case, (requests,), torch.int32)
        case.launch(kernels.prepare_inputs_padded_kernel, (3,),
                    case.tensor(list(itertools.accumulate(counts)) or [0], torch.int32),
                    case.tensor(valid or [0], torch.int32), case.tensor(query, torch.int32),
                    indices, rejected, requests, BLOCK_SIZE=block)
        expected = reference.prepare_padded(counts, valid, query)
        case.check(f"indices_{step}", indices, expected[0])
        case.check(f"rejected_{step}", rejected, expected[1])
        case.check(f"index_tail_{step}", index_tail, [-777] * 64)
        case.check(f"rejected_tail_{step}", rejected_tail, [-777] * 64)


def test_greedy_spec_len_one(case, kernels):
    torch = case.torch
    drafts, targets, bonuses = [3, 4, -1, 6, 9], [3, 8, 2, 7, 9], [20, 21, 22, 23, 24]
    uniforms, rates = [0.9, 0.1, 0.1, 0.4, 0.6], [0.5]
    for step, (synthetic, block) in enumerate(((False, 2), (True, 2), (True, 4), (False, 4))):
        out, tail = guarded(case, (5, 2), torch.int32)
        desc = case.launch(kernels.rejection_greedy_sample_spec_len_1_triton, ((5 + block - 1) // block,),
                           out, case.tensor(drafts, torch.int32), case.tensor(targets, torch.int64),
                           case.tensor(bonuses, torch.int32), 5,
                           case.tensor(uniforms, torch.float32) if synthetic else None,
                           case.tensor(rates, torch.float32) if synthetic else None,
                           SYNTHETIC_MODE=synthetic, BLOCK_SIZE=block)
        case.static(desc, SYNTHETIC_MODE=synthetic)
        expected = reference.greedy_sample([[x] for x in drafts], [[x] for x in targets], bonuses,
                                           uniforms, rates, [True] * 5, synthetic, 1)
        case.check(f"greedy_{step}", out, expected)
        case.check(f"tail_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("with_greedy_mask", [False, True])
def test_greedy_variable_lengths(case, kernels, with_greedy_mask):
    torch = case.torch
    drafts = [[1, 2, 3], [], [4], [5, -1], [7, 8, 9]]
    targets = [[1, 6, 3], [], [4], [5, 4], [7, 8, 9]]
    counts = list(map(len, drafts))
    bonuses = list(range(20, 25))
    uniforms = [0.1, 0.8, 0.1, 0.6, 0.1, 0.2, 0.1, 0.2, 0.3]
    rates = [0.5, 0.5, 0.5]
    enabled = [True, True, False, True, True] if with_greedy_mask else [True] * 5
    for step, (synthetic, block) in enumerate(((False, 2), (True, 2), (True, 4), (False, 4))):
        out, tail = guarded(case, (5, 4), torch.int32)
        desc = case.launch(kernels.rejection_greedy_sample_triton, ((5 + block - 1) // block,),
                           out, case.tensor(list(itertools.accumulate(counts)), torch.int32),
                           case.tensor(list(itertools.chain.from_iterable(drafts)), torch.int32),
                           case.tensor(list(itertools.chain.from_iterable(targets)), torch.int64),
                           case.tensor(bonuses, torch.int32),
                           case.tensor(enabled, torch.int32) if with_greedy_mask else None, 5, 3,
                           case.tensor(uniforms, torch.float32) if synthetic else None,
                           case.tensor(rates, torch.float32) if synthetic else None,
                           SYNTHETIC_MODE=synthetic, BLOCK_SIZE=block)
        case.static(desc, SYNTHETIC_MODE=synthetic)
        expected = reference.greedy_sample(drafts, targets, bonuses, uniforms, rates, enabled, synthetic, 3)
        case.check(f"greedy_{step}", out, expected)
        case.check(f"tail_{step}", tail, [-777] * 64)


def test_sample_recovered_tokens(case, kernels):
    torch = case.torch
    counts, drafts = [2, 0, 1], [3, 5, 7]
    global_vocab, selected_vocab = 70, 35
    # Rank candidates deliberately differs from global token order.
    ids = [(i * 2 + 3) % global_vocab for i in range(selected_vocab)]
    for step, (reduce, no_draft, block) in enumerate((
        (False, True, 32), (False, False, 32), (False, False, 64),
        (True, False, 32), (True, True, 32), (True, True, 64),
    )):
        vocab = selected_vocab if reduce else global_vocab
        target = [[0.25 / (vocab - 2)] * vocab for _ in drafts]
        # Cross a tile boundary and create a tie: the earliest candidate wins.
        for row in target:
            row[1] = row[33] = 0.375
        draft_probs = None if no_draft else [[1 / global_vocab] * global_vocab for _ in drafts]
        indices = [ids] * len(drafts) if reduce else None
        noise = [[1.0] * vocab for _ in counts]
        if reduce:
            for row in noise:
                row[0], row[2], row[4] = 0.0, float("inf"), float("nan")
        out, tail = guarded(case, (len(drafts),), torch.int64)
        desc = case.launch(kernels.sample_recovered_tokens_kernel, (len(counts), max(counts)),
                           out, case.tensor(list(itertools.accumulate(counts)), torch.int32),
                           case.tensor(drafts, torch.int32),
                           None if no_draft else case.tensor(draft_probs, torch.float32),
                           case.tensor(target, torch.float32),
                           case.tensor(indices, torch.int64) if reduce else None,
                           case.tensor(noise, torch.float32), vocab, global_vocab,
                           NO_DRAFT_PROBS=no_draft, ENABLE_REDUCE_SAMPLING=reduce,
                           SUB_BLOCK=block, VOCAB_BLOCK_SIZE=block)
        case.static(desc, NO_DRAFT_PROBS=no_draft, ENABLE_REDUCE_SAMPLING=reduce)
        case.check(f"recovered_{step}", out,
                   reference.recovered_tokens(counts, drafts, target, draft_probs, indices, noise))
        case.check(f"tail_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("strided", [False, True])
def test_apply_all_penalties(case, kernels, strided):
    torch = case.torch
    rows, vocab, step_stride = 3, 513, 2 if strided else 1
    logits = [[((i + r) % 17 - 8) / 4 for i in range(vocab)] for r in range(rows)]
    prompt = [[i % 3 == 0 for i in range(vocab)] for _ in range(rows)]
    counts = [[(i + r) % 4 for i in range(vocab)] for r in range(rows)]
    repetition, frequency, presence = [2, 0.5, 1], [0.25, -0.5, 0], [0.5, -0.25, 0]
    expected = reference.penalties(logits, prompt, counts, repetition, frequency, presence)
    first = None
    for step, block in enumerate((128, 256, 128)):
        backing, tail = guarded(case, (rows, vocab * step_stride + 16), torch.float32)
        values = backing[:, :vocab * step_stride:step_stride]
        values.copy_(torch.tensor(logits))
        # Masks/counts have their own strides and do not alias the logits.
        masks = case.tensor(prompt, torch.bool)
        bins = case.tensor(counts, torch.int32)
        output_mask = bins > 0
        desc = case.launch(kernels.apply_all_penalties_kernel, (2,), values, masks, output_mask, bins,
                    case.tensor(repetition, torch.float32), case.tensor(frequency, torch.float32),
                    case.tensor(presence, torch.float32), rows, vocab,
                    values.stride(0), values.stride(1), masks.stride(0), masks.stride(1),
                    output_mask.stride(0), output_mask.stride(1), bins.stride(0), bins.stride(1),
                    BLOCK_SIZE=block)
        all_expected = torch.full(backing.shape, -777.0)
        all_expected[:, :vocab * step_stride:step_stride] = torch.tensor(expected)
        case.check(f"penalties_{step}", backing, all_expected)
        case.check(f"tail_{step}", tail, [-777] * 64)

        if step == 0:
            first = desc
        elif step == 1:
            case.same_binary(first, desc)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_swiglustep_limit(case, kernels, dtype_name):
    torch = case.torch
    dtype = getattr(torch, dtype_name)
    rows, half, cores = 7, 64, 4
    cpu = ((torch.arange(rows * half * 2).reshape(rows, half * 2) % 61 - 30) / 2).to(dtype)
    inputs = cpu.to(case.device)
    first = None
    for step, limit in enumerate((7.0, 2.0, 0.5, 7.0)):
        out, tail = guarded(case, (rows, half), dtype)
        desc = case.launch(kernels._swiglustep_kernel, (cores,), inputs, out, rows,
                           TOTAL_COLS=2 * half, HALF_COLS=half, LIMIT=limit, NUM_CORES=cores,
                           multibuffer=True)
        gate, up = cpu[:, :half].float(), cpu[:, half:].float()
        expected = (gate * torch.sigmoid(gate)).clamp(max=limit) * up.clamp(-limit, limit)
        case.check(f"swiglustep_{step}", out, expected.to(dtype), rtol=0.008, atol=0.008)
        case.check(f"tail_{step}", tail, [-777] * 64)
        case.static(desc, TOTAL_COLS=2 * half, HALF_COLS=half)
        case.dynamic(desc, "LIMIT")
        if first is None:
            first = desc
        else:
            case.same_binary(first, desc)


@pytest.mark.parametrize("group_dtype", ["int32", "int64"])
def test_swiglu_quant_modes(case, kernels, group_dtype):
    torch = case.torch
    rows, half, cores = 5, 64, 3
    # sigmoid(32) rounds to 1 in FP32. Dyadic inputs pin int8 rounding cases.
    gate = torch.full((rows, half), 32.0)
    up = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 1.0] * (half // 8)).repeat(rows, 1)
    cpu = torch.cat((gate, up), dim=1)
    inputs = cpu.to(case.device)
    for step, (group_type, quant, block, maximum) in enumerate((
        (0, True, 32, 127), (1, True, 32, 127), (1, False, 64, 127),
        (0, False, 32, 127), (0, True, 64, 63),
    )):
        groups = [2, 0, 3] if group_type else [2, 2, 5]
        out, tail = guarded(case, (rows, half), torch.int8 if quant else torch.float32, fill=-77)
        scale, scale_tail = guarded(case, (rows,), torch.float32)
        desc = case.launch(kernels._swiglu_quant_kernel, (cores,), inputs,
                           case.tensor(groups, getattr(torch, group_dtype)), out, scale,
                           TOTAL_COLS=2 * half, HALF_COLS=half, COL_BLOCK_SIZE=block,
                           NUM_EXPERTS=3, NUM_EXPERTS_ALGIN=16 if group_dtype == "int32" else 8,
                           GROUP_LIST_TYPE=group_type, NUM_CORES=cores, DTYPE_MAX=maximum,
                           SCALE=quant, multibuffer=True)
        values = gate * torch.sigmoid(gate) * up
        if quant:
            scales = values.abs().amax(dim=1) / maximum
            expected = torch.trunc(values / scales[:, None]).clamp(-128, 127).to(torch.int8)
            case.check(f"scale_{step}", scale, scales, rtol=2e-6, atol=2e-6)
        else:
            expected = values
            case.check(f"scale_{step}", scale, [-777] * rows)
        case.check(f"swiglu_quant_{step}", out, expected)
        case.check(f"tail_{step}", tail, [-77] * 64)
        case.check(f"scale_tail_{step}", scale_tail, [-777] * 64)
        case.static(desc, GROUP_LIST_TYPE=group_type, SCALE=quant, TOTAL_COLS=2 * half, HALF_COLS=half)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_triton_rope_layouts(case, kernels, partial, dtype_name):
    torch = case.torch
    dtype = getattr(torch, dtype_name)
    tokens, q_heads, k_heads, dim = 3, 5, 3, 32
    rope_dim = 16 if partial else dim
    half = rope_dim // 2
    positions = [2, 0, 1]
    cos_cache = [[0.5 + p / 8 + j / 64 for j in range(half)] for p in range(tokens)]
    sin_cache = [[0.125 + p / 16 - j / 128 for j in range(half)] for p in range(tokens)]
    combined = case.tensor([c + s for c, s in zip(cos_cache, sin_cache)], torch.float32)
    cos = case.tensor([cos_cache[p] for p in positions], torch.float32)
    sin = case.tensor([sin_cache[p] for p in positions], torch.float32)
    pos = case.tensor(positions, torch.int64)
    configs = [(2, True, False), (4, True, False), (4, False, False), (2, False, True), (2, True, True)]
    for step, (block, neox, packed) in enumerate(configs):
        buffers = []
        for heads in (q_heads, k_heads):
            backing, tail = guarded(case, (tokens, heads * dim + 16), dtype)
            cpu = ((torch.arange(tokens * heads * dim).reshape(tokens, heads * dim) % 31 - 15) / 8).to(dtype)
            backing[:, :heads * dim].copy_(cpu)
            buffers.append((backing, tail, cpu))
        q, k = buffers[0][0], buffers[1][0]
        desc = case.launch(kernels._triton_rope, (2,), q, q.stride(0), k, k.stride(0),
                           None if packed else cos, half, None if packed else sin, half,
                           combined if packed else None, rope_dim, pos if packed else None, tokens,
                           n_qh=q_heads, n_kh=k_heads, hd=dim, rope_dim=rope_dim, pad_rope_dim=rope_dim,
                           BLOCK_SIZE_HEAD=block, IS_NEOX_STYLE=neox, USE_COS_SIN=packed)
        case.static(desc, IS_NEOX_STYLE=neox, USE_COS_SIN=packed, n_qh=q_heads, n_kh=k_heads,
                    hd=dim, rope_dim=rope_dim)
        for name, heads, (actual, tail, cpu) in zip(("q", "k"), (q_heads, k_heads), buffers):
            expected = torch.full(actual.shape, -777, dtype=dtype)
            for row in range(tokens):
                for head in range(heads):
                    value = cpu[row, head * dim:(head + 1) * dim].tolist()
                    rotated = reference.rotary(value, cos_cache[positions[row]], sin_cache[positions[row]], neox)
                    expected[row, head * dim:(head + 1) * dim] = torch.tensor(rotated, dtype=dtype)
            case.check(f"rope_{name}_{step}", actual, expected)
            case.check(f"tail_{name}_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_split_qkv_simt(case, kernels, partial, bias):
    torch = case.torch
    batch, heads, dim, cores = 5, 4, 32, 2
    hidden, rope_dim = heads * dim, 16 if partial else dim
    half = rope_dim // 2
    cpu = ((torch.arange(batch * hidden * 3).reshape(batch, hidden * 3) % 41 - 20) / 32).float()
    inputs = cpu.to(case.device)
    weights = torch.linspace(0.5, 1.0, dim)
    biases = torch.arange(dim, dtype=torch.float32) / 64 - 0.25
    weight, bias_ptr = weights.to(case.device), biases.to(case.device) if bias else None
    cache = [[0.5 + row / 16] * half + [0.25 - row / 32] * half for row in range(batch)]
    precomputed = case.tensor(cache, torch.float32)
    configs = [(2, 1e-5), (2, 0.03125), (4, 0.03125), (2, 1e-5)]
    first = None
    for step, (tile, eps) in enumerate(configs):
        buffers = [guarded(case, (batch, hidden), torch.float32) for _ in range(3)]
        q, k, v = [value[0] for value in buffers]
        desc = case.launch(kernels.split_qkv_rmsnorm_rope_simt_kernel, (cores,),
                           inputs, q, k, v, weight, bias_ptr, weight, bias_ptr, precomputed, batch,
                           q_hidden_size=hidden, kv_hidden_size=hidden, total_hidden_size=hidden * 3,
                           eps=eps, BIAS=bias, HEAD_DIM=dim, ROPE_DIM=rope_dim, HALF_ROPE_DIM=half,
                           IS_PARTIAL_ROPE=partial, num_vectorcore=cores,
                           batch_size_per_iter_per_vec=tile, v_batch_size_per_iter_per_vec=2,
                           qk_head_nums_per_iter_per_vec=tile * heads * 2,
                           q_head_num=heads, kv_head_num=heads, qk_head_num_sum=heads * 2,
                           force_simt_only=True)
        case.static(desc, BIAS=bias, IS_PARTIAL_ROPE=partial, HEAD_DIM=dim, ROPE_DIM=rope_dim)
        constants = desc["constants"]
        pair = (constants["batch_size_per_iter_per_vec"], constants["qk_head_nums_per_iter_per_vec"])
        assert pair in {(t, t * heads * 2) for t, _ in configs[:step + 1]}
        for name, actual, values in (("q", q, cpu[:, :hidden]), ("k", k, cpu[:, hidden:2 * hidden])):
            values = values.reshape(batch, heads, dim)
            normalized = values / torch.sqrt(values.square().mean(-1, keepdim=True) + eps) * weights
            if bias:
                normalized += biases
            expected = [[reference.rotary(head.tolist(), cache[row][:half], cache[row][half:])
                         for head in normalized[row]] for row in range(batch)]
            case.check(f"{name}_{step}", actual, torch.tensor(expected).reshape(batch, hidden),
                       rtol=3e-5, atol=3e-5)
        case.check(f"v_{step}", v, cpu[:, 2 * hidden:])
        for name, (_, tail) in zip(("q", "k", "v"), buffers):
            case.check(f"tail_{name}_{step}", tail, [-777] * 64)
        case.dynamic(desc, "eps")
        if step == 0:
            first = desc
        elif step == 1:
            case.same_binary(first, desc)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_split_qkv_mrope(case, kernels, partial, bias):
    torch = case.torch
    tokens, q_heads, k_heads, dim = 5, 4, 2, 32
    q_size, kv_size = q_heads * dim, k_heads * dim
    rope_dim = 16 if partial else dim
    half = rope_dim // 2
    sections = (half // 2, half // 4, half // 4)
    dtype = torch.bfloat16
    q_cpu = ((torch.arange(tokens * q_size).reshape(tokens, q_heads, dim) % 29 - 14) / 16).to(dtype)
    k_cpu = ((torch.arange(tokens * kv_size).reshape(tokens, k_heads, dim) % 23 - 11) / 16).to(dtype)
    v_cpu = (torch.arange(tokens * kv_size).reshape(tokens, kv_size) / 8).to(dtype)
    gate_cpu = (torch.arange(tokens * q_size).reshape(tokens, q_heads, dim) / 8 + 10).to(dtype)
    weight_cpu = torch.full((dim,), 0.75, dtype=dtype)
    bias_cpu = torch.full((dim,), 0.125, dtype=dtype)
    weight = weight_cpu.to(case.device)
    bias_ptr = bias_cpu.to(case.device) if bias else None
    cache = [[[0.5 + axis / 8 + row / 32] * half + [0.125 + axis / 16] * half
              for row in range(tokens)] for axis in range(3)]
    cache_ptr = case.tensor(cache, dtype)
    configs = [(1e-5, False, False), (0.03125, False, False),
               (0.03125, True, False), (0.03125, True, True), (1e-5, False, True)]
    first = None
    for step, (eps, interleaved, gate) in enumerate(configs):
        q_gate = torch.cat((q_cpu, gate_cpu), dim=-1).reshape(tokens, 2 * q_size) if gate else q_cpu.reshape(tokens, q_size)
        inputs = torch.cat((q_gate, k_cpu.reshape(tokens, kv_size), v_cpu), dim=-1).to(case.device)
        buffers = [guarded(case, (tokens, size), dtype) for size in (q_size, kv_size, kv_size, q_size)]
        q, k, v, gate_out = [pair[0] for pair in buffers]
        desc = case.launch(kernels.split_qkv_rmsnorm_mrope_kernel, (3,),
                           inputs, weight, bias_ptr, weight, bias_ptr, cache_ptr, q, k, v, gate_out,
                           tokens, 2, 2, 1, num_q_heads=q_heads, num_kv_heads=k_heads,
                           head_size=dim, q_size=q_size, kv_size=kv_size, eps=eps,
                           mrope_section_t=sections[0], mrope_section_h=sections[1], mrope_section_w=sections[2],
                           has_bias=bias, is_interleaved=interleaved, rope_dim=rope_dim, half_rope_dim=half,
                           IS_PARTIAL_ROPE=partial, gate_size=q_size if gate else 0)
        case.static(desc, has_bias=bias, is_interleaved=interleaved, IS_PARTIAL_ROPE=partial,
                    gate_size=q_size if gate else 0, head_size=dim, num_q_heads=q_heads, num_kv_heads=k_heads)
        channels = reference.mrope_channels(half, sections, interleaved)
        for name, actual, values in (("q", q, q_cpu), ("k", k, k_cpu)):
            values = values.float()
            normalized = values / torch.sqrt(values.square().mean(-1, keepdim=True) + eps) * weight_cpu.float()
            if bias:
                normalized += bias_cpu.float()
            expected = []
            for row in range(tokens):
                cos = [cache[axis][row][j] for j, axis in enumerate(channels)]
                sin = [cache[axis][row][half + j] for j, axis in enumerate(channels)]
                expected.append([reference.rotary(head.tolist(), cos, sin) for head in normalized[row]])
            case.check(f"{name}_{step}", actual, torch.tensor(expected).reshape(actual.shape).to(dtype),
                       rtol=0.016, atol=0.008)
        case.check(f"v_{step}", v, v_cpu)
        case.check(f"gate_{step}", gate_out, gate_cpu.reshape(tokens, q_size) if gate else torch.full((tokens, q_size), -777))
        for name, (_, tail) in zip(("q", "k", "v", "gate"), buffers):
            case.check(f"tail_{name}_{step}", tail, [-777] * 64)
        case.dynamic(desc, "eps")
        if step == 0:
            first = desc
        elif step == 1:
            case.same_binary(first, desc)
