import os

batch_factor = int(os.getenv("BATCH_SIZE", 2))
seq_q = 212 * batch_factor

from easydict import EasyDict as edict

import torch
import torch.nn as nn

from torch._dynamo.testing import rand_strided


NUM_HEADS = 16
KV_LEN = 8000
MASK_SCALE = 10000.0


class VectorMaskedSoftmax(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

    def forward(self, inputs):
        lhs = inputs[0]
        rhs = inputs[1]
        mask = inputs[2]

        # modelD batch2x kernel_details.csv rows 109-116:
        # sub -> rsub -> cast -> mul -> sub -> softmax -> cast -> mul
        score_delta = torch.ops.aten.sub.Tensor(lhs, rhs)
        inverted_mask = 1 - mask
        mask_bias = torch.ops.aten.mul.Tensor(
            inverted_mask.to(torch.float32), MASK_SCALE
        )
        masked_scores = torch.ops.aten.sub.Tensor(score_delta, mask_bias)
        probs = torch.softmax(masked_scores, dim=-1)
        return torch.ops.aten.mul.Tensor(probs, mask.to(torch.float32))


def get_input_data(configs):
    lhs = rand_strided(
        (seq_q, NUM_HEADS, KV_LEN),
        (NUM_HEADS * KV_LEN, KV_LEN, 1),
        device=configs.device,
        dtype=torch.float32,
    )
    rhs = rand_strided(
        (seq_q, NUM_HEADS, KV_LEN),
        (NUM_HEADS * KV_LEN, KV_LEN, 1),
        device=configs.device,
        dtype=torch.float32,
    )
    mask = torch.randint(
        0,
        2,
        (seq_q, 1, KV_LEN),
        device=configs.device,
        dtype=torch.int32,
    )
    return (lhs, rhs, mask)


if __name__ == "__main__":
    configs = edict(
        {
            "model": "VectorMaskedSoftmax",
            "is_compile_mode": True,
            "device": "npu",
        }
    )

    inputs = get_input_data(configs)
    mod = VectorMaskedSoftmax(configs)

    # performance
    import json

    from bench_utils import profile

    eager_fn = lambda: mod(inputs)
    eager_res = profile(eager_fn)
    eager_perf_dump_file = os.path.join(
        "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{batch_factor}-eager-perf.json"
    )
    with open(eager_perf_dump_file, "w", encoding="utf-8") as f:
        json.dump(eager_res, f, ensure_ascii=False, indent=4)

    mod = torch.compile(mod, dynamic=False)
    mod(inputs)

    fn = lambda: mod(inputs)
    res = profile(fn)
    perf_dump_file = os.path.join(
        "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{batch_factor}-perf.json"
    )
    with open(perf_dump_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    eager_total_us = eager_res.get("__total_us__", 0.0)
    compile_total_us = res.get("__total_us__", 0.0)
    if compile_total_us:
        print(
            f"eager total: {eager_total_us:.3f} us, "
            f"compile total: {compile_total_us:.3f} us, "
            f"speedup: {eager_total_us / compile_total_us:.3f}x"
        )
