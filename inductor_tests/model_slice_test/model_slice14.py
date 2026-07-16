import json
import os
import sys

from pathlib import Path

import torch
import torch.nn as nn


repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.append(repo_root)
from TestToolkits.inductor_tests.bench_utils import profile, resolve_device


TARGET_DEVICE = resolve_device()
SOFTMAX_SHAPE = (76800, 150)


class SoftmaxModel(nn.Module):
    def forward(self, inputs):
        return torch.nn.functional.softmax(inputs, dim=-1)


def get_input_data():
    return torch.randn(SOFTMAX_SHAPE, dtype=torch.float32, device=TARGET_DEVICE)


if __name__ == "__main__":
    with torch.no_grad():
        inputs = get_input_data()
        mod = SoftmaxModel()
        eager_output = mod(inputs)
        eager_fn = lambda: mod(inputs)
        eager_res = profile(eager_fn, device=TARGET_DEVICE)
        eager_perf_dump_file = os.path.join(
            "./log", "model_slice14-76800-eager-perf.json"
        )
        with open(eager_perf_dump_file, "w", encoding="utf-8") as file:
            json.dump(eager_res, file, ensure_ascii=False, indent=4)

        mod = torch.compile(mod, dynamic=False)
        compiled_output = mod(inputs)
        torch.testing.assert_close(eager_output, compiled_output)
        fn = lambda: mod(inputs)
        res = profile(fn, device=TARGET_DEVICE)
        perf_dump_file = os.path.join("./log", "model_slice14-76800-perf.json")
        with open(perf_dump_file, "w", encoding="utf-8") as file:
            json.dump(res, file, ensure_ascii=False, indent=4)

        eager_total_us = eager_res.get("__total_us__", 0.0)
        compile_total_us = res.get("__total_us__", 0.0)
        if compile_total_us:
            print(
                f"device: {TARGET_DEVICE}, timing: {res['__backend__']}, "
                f"eager total: {eager_total_us:.3f} us, "
                f"compile total: {compile_total_us:.3f} us, "
                f"speedup: {eager_total_us / compile_total_us:.3f}x"
            )
