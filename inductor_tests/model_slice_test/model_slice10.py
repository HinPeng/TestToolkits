import os
bs = int(os.getenv('BATCH_SIZE', 128))
numel = bs * 16

from easydict import EasyDict as edict

import torch
import torch.nn as nn

from torch._dynamo.testing import rand_strided


class MulSilu(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        
    def forward(self, inputs):
        view_145 = inputs[0]
        mm_37 = inputs[1]
        sigmoid_4: "f32[1, 2048, 8192]" = torch.ops.aten.sigmoid.default(view_145)
        mul_39: "f32[1, 2048, 8192]" = torch.ops.aten.mul.Tensor(view_145, sigmoid_4);  view_145 = sigmoid_4 = None
        view_147: "f32[1, 2048, 8192]" = torch.ops.aten.view.default(mm_37, [1, numel, 8192]);  mm_37 = None
        mul_40: "f32[1, 2048, 8192]" = torch.ops.aten.mul.Tensor(mul_39, view_147);  mul_39 = view_147 = None
        return mul_40
        

def get_input_data(configs):
    buf113 = rand_strided((1, numel, 8192), (numel*8192, 8192, 1), device='npu', dtype=torch.float32)
    buf112 = rand_strided((numel, 8192), (8192, 1), device='npu', dtype=torch.float32)
    return (buf113, buf112)


if __name__ == "__main__":
    configs = edict({
        "model": "MulSilu",     
        "is_compile_mode": True,
        "device": "npu",
    })
    
    inputs = get_input_data(configs)
    mod = MulSilu(configs)

    # performance
    import json
    import sys
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from TestToolkits.inductor_tests.bench_utils import profile

    eager_fn = lambda: mod(inputs)
    eager_res = profile(eager_fn)
    eager_perf_dump_file = os.path.join(
        "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{bs}-eager-perf.json"
    )
    with open(eager_perf_dump_file, 'w', encoding='utf-8') as f:
        json.dump(eager_res, f, ensure_ascii=False, indent=4)

    mod = torch.compile(mod, dynamic=False)
    mod(inputs)

    fn = lambda: mod(inputs)
    res = profile(fn)
    perf_dump_file = os.path.join(
        "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{bs}-perf.json"
    )
    with open(perf_dump_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

    eager_total_us = eager_res.get("__total_us__", 0.0)
    compile_total_us = res.get("__total_us__", 0.0)
    if compile_total_us:
        print(
            f"eager total: {eager_total_us:.3f} us, "
            f"compile total: {compile_total_us:.3f} us, "
            f"speedup: {eager_total_us / compile_total_us:.3f}x"
        )
