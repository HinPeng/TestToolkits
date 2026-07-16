import os
bs = int(os.getenv('BATCH_SIZE', 128))

from easydict import EasyDict as edict

import torch

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.append(repo_root)
from TestToolkits.inductor_tests.bench_utils import profile, resolve_device

TARGET_DEVICE = resolve_device()
import torch.nn as nn

from torch._dynamo.testing import rand_strided


class EmbeddingSum(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        
    def forward(self, inputs):
        arg0_1 = inputs[0]
        arg2_1 = inputs[1]
        view: "i64[128, 4000]" = torch.ops.aten.view.default(arg0_1, [-1, 4000]); arg0_1 = None
        embedding: "f32[128, 4000, 128]" = torch.ops.aten.embedding.default(arg2_1, view); arg2_1 = view = None
        sum_1: "f32[128, 128]" = torch.ops.aten.sum.dim_IntList(embedding, [1]); embedding = None
        return sum_1
        

def get_input_data(configs):
    arg0_1 = torch.randint(0, 9000, size=(bs, 4000), device=configs.device, dtype=torch.int32)
    arg2_1 = rand_strided((9000, bs), (bs, 1), device=configs.device, dtype=torch.float32)
    return (arg0_1, arg2_1)


if __name__ == "__main__":
    configs = edict({
        "model": "EmbeddingSum",
        "is_compile_mode": True,
        "device": TARGET_DEVICE,
    })
    
    inputs = get_input_data(configs)
    mod = EmbeddingSum(configs)

    # performance
    import json
    eager_fn = lambda: mod(inputs)
    eager_res = profile(eager_fn, device=TARGET_DEVICE)
    eager_perf_dump_file = os.path.join(
        "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{bs}-eager-perf.json"
    )
    with open(eager_perf_dump_file, 'w', encoding='utf-8') as f:
        json.dump(eager_res, f, ensure_ascii=False, indent=4)

    mod = torch.compile(mod, dynamic=False)
    mod(inputs)

    fn = lambda: mod(inputs)
    res = profile(fn, device=TARGET_DEVICE)
    perf_dump_file = os.path.join(
        "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{bs}-perf.json"
    )
    with open(perf_dump_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    eager_total_us = eager_res.get("__total_us__", 0.0)
    compile_total_us = res.get("__total_us__", 0.0)
    if compile_total_us:
        print(
            f"device: {TARGET_DEVICE}, timing: {res['__backend__']}, eager total: {eager_total_us:.3f} us, "
            f"compile total: {compile_total_us:.3f} us, "
            f"speedup: {eager_total_us / compile_total_us:.3f}x"
        )
