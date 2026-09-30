"""Small CPU oracles, deliberately independent of Triton tiling and dispatch."""

import math


def slot_mapping(lengths, positions, tables, block_sizes, circular, capacity, pad):
    outputs = []
    for table, block_size, is_circular in zip(tables, block_sizes, circular):
        output = []
        start = 0
        for req, length in enumerate(lengths):
            for pos in positions[start:start + length]:
                if is_circular and pos < 0:
                    output.append(pad)
                else:
                    index = 0 if is_circular else pos // block_size
                    output.append(table[req][index] * block_size + pos % block_size)
            start += length
        outputs.append(output + [pad] * (capacity - len(output)))
    return outputs


def metadata(query, seq_lens, start, end, compute_start):
    prefix, lengths, starts = [0], [], []
    for left, right, seq in zip(query, query[1:], seq_lens):
        lo, hi = max(start, min(left, end)), max(start, min(right, end))
        count = hi - lo
        prefix.append(prefix[-1] + count)
        lengths.append(max(seq - (right - hi), 0) if count > 0 and seq > 0 else 0)
        starts.append(seq - (right - left) if compute_start else -777)
    return prefix, lengths, starts


def bin_counts(tokens, vocab, rank=0):
    outputs = []
    for row in tokens:
        counts = [0] * vocab
        for token in row:
            index = token - rank * vocab
            if 0 <= index < vocab:
                counts[index] += 1
        outputs.append(counts)
    return outputs


def ngram(row, length, sampled, discard, vocab, min_n, max_n, k):
    row = list(row)
    filtered = [-1 if discard or value == -1 or value >= vocab else value for value in sampled]
    raw = sum(value != -1 for value in filtered)
    count = min(raw, max(0, len(row) - length))
    next_token = filtered[count - 1] if count else row[max(0, length - 1)]
    row[length:length + count] = filtered[:count]
    total = length + count
    best = None
    if count:
        for size in range(min(max_n, total), min_n - 1, -1):
            for pos in range(total - size):
                if row[pos:pos + size] == row[total - size:total]:
                    best = pos + size
                    break
            if best is not None:
                break
    draft = [] if best is None else row[best:min(best + k, total)]
    return row, next_token, draft + [-1] * (k - len(draft)), len(draft), raw


def rejection(drafts, target, draft_probs, uniforms, rates, greedy, synthetic, entropy):
    outputs = []
    for req in range(len(greedy)):
        output = [-777] * 3
        if not greedy[req]:
            rejected = False
            for pos in range(2):
                token_index = req * 2 + pos
                token = drafts[token_index]
                if synthetic:
                    accept = token >= 0 and uniforms[token_index] < rates[pos]
                elif token < 0:
                    accept = False
                else:
                    p = target[token_index][token]
                    q = 1.0 if draft_probs is None else draft_probs[token_index][token]
                    threshold = 1.0
                    if entropy:
                        h = -sum(v * math.log(v + 1e-10) for v in target[token_index])
                        threshold = min(math.exp(-h * 0.4), 0.95)
                    accept = q > 0 and p / q >= uniforms[token_index] * threshold
                output[pos] = token if accept else 50 + token_index
                if not accept:
                    rejected = True
                    break
            if not rejected:
                output[2] = 60 + req
        outputs.append(output)
    return outputs


def prepare_padded(counts, valid_counts, query):
    rejected = [drafts + 1 - valid if drafts else 0 for drafts, valid in zip(counts, valid_counts)]
    indices = [end - 1 - rejected_count for end, rejected_count in zip(query[1:], rejected)]
    return indices, rejected


def greedy_sample(drafts, targets, bonuses, uniforms, rates, enabled, synthetic, max_spec):
    output = []
    cursor = 0
    for req, (row, target) in enumerate(zip(drafts, targets)):
        result = [-777] * (max_spec + 1)
        if enabled[req]:
            accepted_all = True
            for pos, (draft, wanted) in enumerate(zip(row, target)):
                accepted = (draft >= 0 and uniforms[cursor + pos] < rates[pos]) if synthetic else draft == wanted
                result[pos] = draft if synthetic and accepted else wanted
                if not accepted:
                    accepted_all = False
                    break
            if accepted_all:
                result[len(row)] = bonuses[req]
        output.append(result)
        cursor += len(row)
    return output


def recovered_tokens(counts, drafts, target, draft_probs, indices, noise):
    output = []
    token_index = 0
    for req, count in enumerate(counts):
        for _ in range(count):
            ids = indices[token_index] if indices is not None else range(len(target[token_index]))
            best_id, best_score = -1, -math.inf
            for col, (token, probability) in enumerate(zip(ids, target[token_index])):
                if draft_probs is None:
                    probability = 0 if token == drafts[token_index] else probability
                else:
                    probability = max(probability - draft_probs[token_index][token], 0)
                q = noise[req][col]
                score = probability / q if math.isfinite(q) and q > 0 else -math.inf
                if score > best_score:
                    best_id, best_score = token, score
            output.append(best_id)
            token_index += 1
    return output


def penalties(logits, prompt, counts, repetition, frequency, presence):
    result = []
    for row, values in enumerate(logits):
        output = []
        for col, value in enumerate(values):
            seen = counts[row][col] > 0
            if prompt[row][col] or seen:
                value = value / repetition[row] if value > 0 else value * repetition[row]
            output.append(value - frequency[row] * counts[row][col] - presence[row] * seen)
        result.append(output)
    return result


def rotary(values, cos, sin, neox=True):
    result = list(values)
    half = len(cos)
    for j, (c, s) in enumerate(zip(cos, sin)):
        left, right = (j, j + half) if neox else (2 * j, 2 * j + 1)
        result[left] = values[left] * c - values[right] * s
        result[right] = values[right] * c + values[left] * s
    return result


def mrope_channels(half, sections, interleaved):
    t, h, w = sections
    assert t + h + w == half
    if interleaved:
        return [1 if j % 3 == 1 and j <= 3 * h else
                2 if j % 3 == 2 and j <= 3 * w else 0 for j in range(half)]
    return [0] * t + [1] * h + [2] * w
