"""Same-process invocation sequences against frozen vLLM Ascend JIT kernels."""

import itertools

import pytest

import reference

pytestmark = pytest.mark.npu


def guarded(case, shape, dtype, fill=-777):
    torch = case.torch
    count = 1
    for dim in shape:
        count *= dim
    storage = torch.full((count + 64,), fill, dtype=dtype, device=case.device)
    return storage[:count].reshape(shape), storage[count:]


@pytest.mark.parametrize("blocks", [(1024, 2048), (2048, 1024)])
@pytest.mark.parametrize("grid_size", [1, 3, 7])
def test_muls_add_tile_reuse(case, kernels, blocks, grid_size):
    torch = case.torch
    n = 5000
    # Dyadic inputs make the result exact with or without FMA.
    x_cpu = (torch.arange(n, dtype=torch.float32) % 31 - 15) / 4
    y_cpu = (torch.arange(n, dtype=torch.float32) % 19 - 9) / 8
    x, y = x_cpu.to(case.device), y_cpu.to(case.device)
    first = None
    for step, block in enumerate((*blocks, blocks[1])):
        out, tail = guarded(case, (n,), torch.float32)
        grid_calls = []

        def grid(meta):
            grid_calls.append(meta["BLOCK_SIZE"])
            return (grid_size,)

        desc = case.launch(kernels.muls_add_kernel, grid, x, y, out, 0.5,
                           n, (n + block - 1) // block, BLOCK_SIZE=block)
        assert len(grid_calls) == 1, "the original grid callable must be evaluated once"
        case.check(f"output_{step}", out, x_cpu * 0.5 + y_cpu)
        case.check(f"tail_{step}", tail, [-777] * 64)
        if step == 0:
            first = desc
        elif step == 1:
            # At the first miss only the first requested tile is READY.
            case.same_binary(first, desc)
    case.check("input_x", x, x_cpu)
    case.check("input_y", y, y_cpu)


@pytest.mark.parametrize("n", [1, 31, 1023, 1024, 1025, 4097])
def test_muls_add_boundaries(case, kernels, n):
    torch = case.torch
    x = torch.arange(n, dtype=torch.float32, device=case.device)
    y = torch.ones_like(x)
    for step, block in enumerate((1024, 2048)):
        out, tail = guarded(case, (n,), torch.float32)
        case.launch(kernels.muls_add_kernel, (3,), x, y, out, 0.5,
                    n, (n + block - 1) // block, BLOCK_SIZE=block)
        case.check(f"output_{step}", out, torch.arange(n, dtype=torch.float32) * 0.5 + 1)
        case.check(f"tail_{step}", tail, [-777] * 64)


def test_muls_add_partial_coverage(case, kernels):
    torch = case.torch
    n = 5000
    x = torch.ones(n, device=case.device)
    y = torch.ones_like(x)
    warm, _ = guarded(case, (n,), torch.float32)
    case.launch(kernels.muls_add_kernel, (3,), x, y, warm, 0.5, n, 5, BLOCK_SIZE=1024)
    case.check("warm", warm, [1.5] * n)
    out, tail = guarded(case, (n,), torch.float32)
    desc = case.launch(kernels.muls_add_kernel, (3,), x, y, out, 0.5, n, 1, BLOCK_SIZE=2048)
    case.static(desc, BLOCK_SIZE=2048)
    case.check("partial", out, [1.5] * 2048 + [-777] * (n - 2048))
    case.check("tail", tail, [-777] * 64)


def test_muls_add_exact_alias(case, kernels):
    torch = case.torch
    n = 5000
    cpu = torch.arange(n, dtype=torch.float32) / 4
    y = torch.ones(n, device=case.device)
    for step, block in enumerate((1024, 2048)):
        x, tail = guarded(case, (n,), torch.float32)
        x.copy_(cpu)
        case.launch(kernels.muls_add_kernel, (3,), x, y, x, 0.5,
                    n, (n + block - 1) // block, BLOCK_SIZE=block)
        case.check(f"alias_{step}", x, cpu * 0.5 + 1)
        case.check(f"tail_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("adaptive,parallel", [(False, 1), (False, 2), (True, 1)])
@pytest.mark.parametrize("circular", [False, True])
def test_slot_mapping_dynamic(case, kernels, adaptive, parallel, circular):
    torch = case.torch
    first = None
    sequence = [(3, -1), (4, -7), (1, 1), (0, -13), (3, -(2**31)), (3, 2**31 - 1)]
    for step, (requests, pad) in enumerate(sequence):
        lengths = ([1, 33, 130, 7] if adaptive else [17, 29, 35, 7])[:requests]
        query = [0, *itertools.accumulate(lengths)]
        positions = [(-1 if circular and i == 0 else i + r * 3)
                     for r, length in enumerate(lengths) for i in range(length)]
        capacity = 512
        tables = [[[g * 1000 + r * 128 + b + 1 for b in range(128)]
                   for r in range(4)] for g in range(2)]
        table_tensors = [case.tensor(t, torch.int32) for t in tables]
        outputs_and_tails = [guarded(case, (capacity,), torch.int32) for _ in range(2)]
        outputs = [pair[0] for pair in outputs_and_tails]
        block_sizes = [16, 32]
        args = (
            len(positions), capacity, case.tensor(query, torch.int32),
            case.tensor(positions or [0], torch.int64),
            case.tensor([t.data_ptr() for t in table_tensors], torch.int64),
            case.tensor([t.data_ptr() for t in outputs], torch.int64),
            case.tensor([128, 128], torch.int32), case.tensor(block_sizes, torch.int32),
            case.tensor([circular, circular], torch.int32),
        )
        kwargs = dict(HAS_CIRCULAR=True, PAD_ID=pad, NUM_REQS=requests)
        if adaptive:
            kernel = kernels._compute_slot_mapping_fused_groups_adaptive_kernel
            kwargs.update(SMALL_TILE_BLOCK_SIZE=32, SMALL_BLOCK_TABLE_WINDOW_SIZE=8,
                          LARGE_BLOCK_TABLE_WINDOW_SIZE=128)
            grid = (2 * (requests + 1),)
        else:
            kernel = kernels._compute_slot_mapping_fused_groups_kernel
            kwargs.update(TILE_BLOCK_SIZE=32, PARALLEL_TILES=parallel, BLOCK_TABLE_WINDOW_SIZE=8)
            grid = (2 * (requests * parallel + 1),)
        desc = case.launch(kernel, grid, *args, **kwargs)
        expected = reference.slot_mapping(lengths, positions, tables, block_sizes,
                                          [circular, circular], capacity, pad)
        for group, (out, tail) in enumerate(outputs_and_tails):
            case.check(f"slots_{step}_{group}", out, expected[group])
            case.check(f"tail_{step}_{group}", tail, [-777] * 64)
        case.dynamic(desc, "NUM_REQS", "PAD_ID")
        if first is None:
            first = desc
        else:
            case.same_binary(first, desc)


def test_rms_dim_static(case, kernels):
    torch = case.torch
    for step, dim in enumerate((256, 512, 256)):
        # Extra row stride also detects accidental use of the old DIM.
        rows, stride = 5, dim + 16
        cpu = (torch.arange(rows * stride, dtype=torch.float32).reshape(rows, stride) % 29 + 1) / 16
        x = cpu.to(case.device)
        out, tail = guarded(case, (rows, stride), torch.float32)
        desc = case.launch(kernels.triton_rms_kernel, (1,), x, stride, out,
                           1e-5, rows, DIM=dim, BLOCK_M=2)
        case.static(desc, DIM=dim)
        expected = torch.full((rows, stride), -777.0)
        values = cpu[:, :dim]
        expected[:, :dim] = values * torch.rsqrt((values * values).mean(-1, keepdim=True) + 1e-5)
        case.check(f"rms_{step}", out, expected, rtol=2e-5, atol=2e-5)
        case.check(f"tail_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("compute_start", [False, True])
def test_metadata_capacity(case, kernels, compute_start):
    torch = case.torch
    for step, (requests, block) in enumerate(((63, 64), (63, 128), (65, 128), (63, 64), (64, 64), (1, 128))):
        query = [i * 3 for i in range(requests + 1)]
        seq = [100 + i for i in range(requests)]
        # Include request 65 in the local interval, making stale capacity visible.
        start, end = 2, query[-1] - 1
        prefix, prefix_tail = guarded(case, (requests + 1,), torch.int32)
        prefix.zero_()
        lengths, length_tail = guarded(case, (requests,), torch.int32)
        lengths.zero_()
        starts, start_tail = guarded(case, (requests,), torch.int32)
        desc = case.launch(kernels.build_local_metadata_triton, (1,),
                           case.tensor(query, torch.int32), case.tensor(seq, torch.int32),
                           prefix, lengths, start, end, requests, starts,
                           BLOCK_NUM_REQS=block, COMPUTE_START_POS=compute_start)
        # User contract: no cross-value reuse, even if both capacities suffice.
        case.static(desc, BLOCK_NUM_REQS=block, COMPUTE_START_POS=compute_start)
        expected = reference.metadata(query, seq, start, end, compute_start)
        for name, value, want in zip(("prefix", "lengths", "starts"), (prefix, lengths, starts), expected):
            case.check(f"{name}_{step}", value, want)
        for name, value in zip(("prefix", "length", "start"), (prefix_tail, length_tail, start_tail)):
            case.check(f"{name}_tail_{step}", value, [-777] * 64)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
def test_bincount_external_bound(case, kernels, rank, strided):
    torch = case.torch
    batch, seq, vocab = 2, 300, 32
    tokens = [[(i * 7 + r * 3) % 70 - 2 for i in range(seq)] for r in range(batch)]
    # Last 172 tokens in row 1 are observable in the U2 counterexample.
    cpu = torch.tensor(tokens, dtype=torch.int32)
    if strided:
        backing = torch.full((batch, seq * 2), -1, dtype=torch.int32)
        backing[:, ::2] = cpu
        inputs = backing.to(case.device)[:, ::2]
    else:
        inputs = cpu.to(case.device)
    for step, block in enumerate((128, 256, 128)):
        counts, tail = guarded(case, (batch, vocab), torch.int32)
        counts.zero_()
        total_blocks = batch * ((seq + block - 1) // block)
        desc = case.launch(kernels.token_bin_counts_and_mask_kernel, (3,), inputs,
                           inputs.stride(0), inputs.stride(1), batch, seq, vocab, counts,
                           rank, counts.stride(0), counts.stride(1), total_blocks,
                           SEQ_BLOCK=block, multibuffer=False)
        case.static(desc, SEQ_BLOCK=block)
        case.check(f"counts_{step}", counts, reference.bin_counts(tokens, vocab, rank))
        case.check(f"tail_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("max_seq", [128, 2048])
def test_ngram_static_range(case, kernels, max_seq):
    torch = case.torch
    rows = [[7, 1, 2, 9, 8, 1], [4, 5, 6], [2, 3, 2, 3], [1, 2]]
    lengths = list(map(len, rows))
    rows = [row + [0] * (max_seq - len(row)) for row in rows]
    sampled = [[2, -1, -1, -1], [8, -1, -1, -1], [2, -1, -1, -1], [99, -1, -1, -1]]
    discard = [0, 0, 1, 0]
    for step, min_n in enumerate((2, 3, 2)):
        inputs = case.tensor(rows, torch.int32)
        next_ids, next_tail = guarded(case, (4,), torch.int32)
        drafts, draft_tail = guarded(case, (4, 4), torch.int32)
        valid, valid_tail = guarded(case, (4,), torch.int32)
        raw, raw_tail = guarded(case, (4,), torch.int32)
        desc = case.launch(kernels.ngram_spec_decode_kernel, (2,), inputs,
                           case.tensor(lengths, torch.int32), case.tensor(sampled, torch.int32),
                           case.tensor(discard, torch.int32), next_ids, drafts, valid, raw,
                           max_seq_len=max_seq, max_new_tokens=4, vocab_size=32,
                           min_n=min_n, max_n=4, k=4, batch_size=4)
        case.static(desc, min_n=min_n)
        expected = [reference.ngram(row, length, sample, drop, 32, min_n, 4, 4)
                    for row, length, sample, drop in zip(rows, lengths, sampled, discard)]
        for name, actual, want in zip(("tokens", "next", "draft", "valid", "raw"),
                                      (inputs, next_ids, drafts, valid, raw), zip(*expected)):
            case.check(f"{name}_{step}", actual, want)
        for name, tail in zip(("next", "draft", "valid", "raw"), (next_tail, draft_tail, valid_tail, raw_tail)):
            case.check(f"{name}_tail_{step}", tail, [-777] * 64)


def test_expand_signature_and_integer_width(case, kernels):
    torch = case.torch
    for step, dtype in enumerate((torch.int32, torch.int64, torch.int32)):
        large = 2**40 + 17 if dtype == torch.int64 else 17
        values, counts = [large, -1, 29, 7], [1, 3, 0, 2]
        cumulative = list(itertools.accumulate(counts))
        replace = large + 3
        out, tail = guarded(case, (sum(counts),), dtype)
        desc = case.launch(kernels.expand_kernel, (2,), out, case.tensor(values, dtype),
                           case.tensor(cumulative, torch.int32), -1, replace, 4,
                           MAX_NUM_TOKENS=4, BLOCK_SIZE=2)
        expected = [replace if v == -1 else v for v, n in zip(values, counts) for _ in range(n)]
        case.check(f"expand_{step}", out, expected)
        case.check(f"tail_{step}", tail, [-777] * 64)
        signature = desc["signature"]
        assert signature["input_ptr"] == signature["output_ptr"]
        assert signature["input_ptr"] == ("*i64" if dtype == torch.int64 else "*i32")


@pytest.mark.parametrize("no_draft", [True, False])
@pytest.mark.parametrize("no_original", [True, False])
def test_rejection_modes_and_entropy(case, kernels, no_draft, no_original):
    torch = case.torch
    drafts = [0, 1, 0, 1, 0, -1, 0, 1]
    target = [[0.4] + [0.6 / 63] * 63 for _ in range(8)]
    draft_probs = None if no_draft else [[0.8] + [0.2 / 63] * 63 for _ in range(8)]
    uniforms = [0.6, 0.8, 0.05, 0.05, 0.05, 0.25, 0.2, 0.2]
    rates, greedy = [0.9, 0.1], [0, 0, 0, 1]
    # Same inputs, mode transitions, then a different entropy reduction grouping.
    modes = [(False, False, 32), (True, False, 32), (False, True, 32),
             (False, True, 64), (False, False, 32)]
    for step, (synthetic, entropy, sub_block) in enumerate(modes):
        output, tail = guarded(case, (4, 3), torch.int32)
        desc = case.launch(
            kernels.rejection_random_sample_kernel, (2,), output,
            case.tensor([2, 4, 6, 8], torch.int32), case.tensor(drafts, torch.int32),
            None if no_draft else case.tensor(draft_probs, torch.float32),
            case.tensor(target, torch.float32), None,
            case.tensor([60, 61, 62, 63], torch.int32), case.tensor(list(range(50, 58)), torch.int32),
            case.tensor(uniforms, torch.float32), case.tensor(greedy, torch.int32),
            2, 64, 64, 4,
            None if no_original else case.tensor(target, torch.float32),
            case.tensor(rates, torch.float32) if synthetic else None,
            NO_ORI_TARGET_PROBS=no_original, NO_DRAFT_PROBS=no_draft,
            ENABLE_REDUCE_SAMPLING=False, SYNTHETIC_MODE=synthetic,
            ENTROPY_VERIFY=entropy, BLOCK_SIZE=2, SUB_BLOCK=sub_block,
        )
        case.static(desc, SYNTHETIC_MODE=synthetic, ENTROPY_VERIFY=entropy,
                    NO_DRAFT_PROBS=no_draft, NO_ORI_TARGET_PROBS=no_original)
        if entropy:
            case.static(desc, SUB_BLOCK=sub_block)
        expected = reference.rejection(drafts, target, draft_probs, uniforms, rates, greedy, synthetic, entropy)
        case.check(f"tokens_{step}", output, expected)
        case.check(f"tail_{step}", tail, [-777] * 64)


@pytest.mark.parametrize("change", ["eps", "complete_tiles"])
def test_rope_eps_and_complete_versions(case, kernels, change):
    torch = case.torch
    batch, heads, dim = 4, 8, 32
    hidden = heads * dim
    cpu = ((torch.arange(batch * hidden * 3).reshape(batch, hidden * 3) % 41 - 20) / 32).to(torch.bfloat16)
    input_tensor = cpu.to(case.device)
    weight = torch.ones(dim, dtype=torch.bfloat16, device=case.device)
    positions = case.tensor([0, 1, 2, 3], torch.int64)
    # Nontrivial, exactly representable cos/sin; no padded position lookup.
    cache = torch.tensor([[0.5] * 16 + [0.25] * 16] * 4, dtype=torch.bfloat16)
    cache_device = cache.to(case.device)
    configs = [(2, 32, 1e-5), (2, 32, 0.03125), (2, 32, 1e-3)] if change == "eps" else [
        (2, 32, 1e-5), (4, 64, 1e-5), (2, 32, 1e-5)]
    first = None
    for step, (rows, tiled_heads, eps) in enumerate(configs):
        buffers = [guarded(case, (batch, hidden), torch.bfloat16) for _ in range(3)]
        q, k, v = [pair[0] for pair in buffers]
        desc = case.launch(
            kernels.split_qkv_rmsnorm_rope_kernel, (1,), input_tensor, q, k, v,
            weight, None, weight, None, batch,
            q_hidden_size=hidden, kv_hidden_size=hidden, total_hidden_size=3 * hidden,
            eps=eps, BIAS=False, HEAD_DIM=dim, ROPE_DIM=dim, HALF_ROPE_DIM=dim // 2,
            IS_PARTIAL_ROPE=False, num_vectorcore=1,
            batch_size_per_iter_per_vec=rows, qk_head_nums_per_iter_per_vec=tiled_heads,
            q_head_num=heads, kv_head_num=heads, qk_head_num_sum=heads * 2,
            v_batch_size_per_iter_per_vec=2, positions_gm_ptr=positions,
            cos_sin_cache_gm_ptr=cache_device,
        )
        selected = desc["constants"]
        pair = (selected["batch_size_per_iter_per_vec"], selected["qk_head_nums_per_iter_per_vec"])
        # A complete requested version is legal; a hybrid (2,64)/(4,32) is not.
        assert pair in {(c[0], c[1]) for c in configs[:step + 1]}
        case.static(desc, HEAD_DIM=dim, q_head_num=heads, kv_head_num=heads)
        for name, actual, data in zip(("q", "k"), (q, k), (cpu[:, :hidden], cpu[:, hidden:2 * hidden])):
            values = data.reshape(batch, heads, dim).float()
            normalized = (values / torch.sqrt(values.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
            left, right = normalized[..., :16].float(), normalized[..., 16:].float()
            expected = torch.cat((left * 0.5 - right * 0.25, right * 0.5 + left * 0.25), -1)
            case.check(f"{name}_{step}", actual, expected.reshape(batch, hidden).to(torch.bfloat16),
                       rtol=0.02, atol=0.016)
        case.check(f"v_{step}", v, cpu[:, 2 * hidden:])
        for name, (_, tail) in zip(("q", "k", "v"), buffers):
            # -777 rounds to -776 in BF16, as does the expected cast.
            case.check(f"{name}_tail_{step}", tail, [-777] * 64)
        if change == "eps":
            case.dynamic(desc, "eps")
            if first is None:
                first = desc
            else:
                case.same_binary(first, desc)


def test_renamed_pointwise_reuse(case, probes):
    torch = case.torch
    n = 5000
    a = torch.ones(n, device=case.device)
    b = torch.ones_like(a)
    first = None
    for step, block in enumerate((1024, 2048)):
        out, tail = guarded(case, (n,), torch.float32)
        desc = case.launch(probes.renamed_pointwise, (3,), a, b, out, 0.5, n,
                           (n + block - 1) // block, CHUNK=block)
        case.check(f"renamed_{step}", out, [1.5] * n)
        case.check(f"tail_{step}", tail, [-777] * 64)
        if first is None:
            first = desc
        else:
            case.same_binary(first, desc)


def test_copy_external_bound_without_atomics(case, probes):
    torch = case.torch
    batch, width = 2, 300
    cpu = torch.arange(batch * width, dtype=torch.int32).reshape(batch, width)
    inputs = cpu.to(case.device)
    for step, block in enumerate((128, 256, 128)):
        out, tail = guarded(case, (batch, width), torch.int32)
        limit = batch * ((width + block - 1) // block)
        desc = case.launch(probes.external_bound_copy, (3,), inputs, out, batch, width, limit, CHUNK=block)
        case.static(desc, CHUNK=block)
        case.check(f"copy_{step}", out, cpu)
        case.check(f"tail_{step}", tail, [-777] * 64)
