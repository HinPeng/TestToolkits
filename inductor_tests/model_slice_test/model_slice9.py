import os
bs = int(os.getenv('BATCH_SIZE', 128))  # failed for 1024
numel = bs * 6

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


class LayerNormGetItemAddModel(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        
    def forward(self, inputs):
        native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(inputs[0], [numel], inputs[1], inputs[2], 1e-12)
        getitem_3: "f32[8, 384, 768]" = native_layer_norm_1[0]
        add_4: "f32[8, 384, 768]" = torch.ops.aten.add.Tensor(inputs[3], getitem_3)
        return add_4


def get_input_data(configs):
    add_3: "f32[8, 384, 768]" = rand_strided((8, 384, numel), (8*384*numel, numel, 1), torch.float32, device=configs.device)
    arg17_1: "f32[768]" = rand_strided((numel,), (1,), torch.float32, device=configs.device)
    arg18_1: "f32[768]" = rand_strided((numel,), (1,), torch.float32, device=configs.device)
    view_21: "f32[8, 384, 768]" = rand_strided((8, 384, numel), (8*384*numel, numel, 1), torch.float32, device=configs.device)
    return (add_3, arg17_1, arg18_1, view_21)


if __name__ == "__main__":
    configs = edict({
        "model": "layernorm_getitem_add",
        "is_compile_mode": True,
        "device": TARGET_DEVICE,
    })
    
    inputs = get_input_data(configs)
    mod = LayerNormGetItemAddModel(configs)

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
