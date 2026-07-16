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


class AddSigmoidCatViewSelect(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        
    def forward(self, inputs):
        in_ptr0 = inputs[0]
        in_ptr1 = inputs[1]
        in_ptr2 = inputs[2]
        in_ptr3 = inputs[3]
        add: "f32[128, 1]" = torch.ops.aten.add.Tensor(in_ptr0, in_ptr1)
        sigmoid: "f32[128, 1]" = torch.ops.aten.sigmoid.default(add)

        add_1: "f32[128, 1]" = torch.ops.aten.add.Tensor(in_ptr2, in_ptr3)

        sigmoid_1: "f32[128, 1]" = torch.ops.aten.sigmoid.default(add_1)

        cat_2: "f32[128, 2]" = torch.ops.aten.cat.default([sigmoid, sigmoid_1], -1)

        view_324: "f32[128, 2]" = torch.ops.aten.view.default(cat_2, [-1, 2])

        select: "f32[128]" = torch.ops.aten.select.int(view_324, 1, 0)
        return select
        

def get_input_data(configs):
    in_ptr0 = rand_strided((bs, 1), (1, 1), device=configs.device, dtype=torch.float32)
    in_ptr1 = rand_strided((1, ), (1, ), device=configs.device, dtype=torch.float32)
    in_ptr2 = rand_strided((bs, 1), (1, 1), device=configs.device, dtype=torch.float32)
    in_ptr3 = rand_strided((1, ), (1, ), device=configs.device, dtype=torch.float32)
    return (in_ptr0, in_ptr1, in_ptr2, in_ptr3)


if __name__ == "__main__":
    configs = edict({
        "model": "AddSigmoidCatViewSelect",
        "is_compile_mode": True,
        "device": TARGET_DEVICE,
    })
    
    inputs = get_input_data(configs)
    mod = AddSigmoidCatViewSelect(configs)

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
