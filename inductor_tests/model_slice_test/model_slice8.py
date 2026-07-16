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


class EqWhere(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        
    def forward(self, inputs):
        arg34_1 = inputs[0]
        eq: "b8[4096, 50]" = torch.ops.aten.eq.Scalar(arg34_1, -1)
        full_default: "i64[4096, 50]" = torch.ops.aten.full.default([bs*32, 50], 0, dtype = torch.int32, layout = torch.strided, device =  self.configs.device, pin_memory = False)
        where: "i64[4096, 50]" = torch.ops.aten.where.self(eq, full_default, arg34_1);  eq = full_default = arg34_1 = None
        return where
        

def get_input_data(configs):
    numel = bs * 32
    arg34_1 = rand_strided((numel, 50), (50, 1), device=configs.device, dtype=torch.int32)
    return (arg34_1,)


if __name__ == "__main__":
    configs = edict({
        "model": "EqWhere",
        "is_compile_mode": True,
        "device": TARGET_DEVICE,
    })
    
    inputs = get_input_data(configs)
    mod = EqWhere(configs)

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
