import os

os.environ.setdefault("INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE", "1")
os.environ.setdefault("INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES", "pointwise")

import torch
import torch_npu  # noqa: F401


torch._inductor.config.force_disable_caches = True
torch._dynamo.config.recompile_limit = 4096

CURRENT_DEVICE = "npu"
TEST_LENGTHS = (128, 1024, 8192)


def eager(a, b):
    return a + b


def make_inputs(length):
    a = torch.randn(length, device=CURRENT_DEVICE)
    b = torch.randn(length, device=CURRENT_DEVICE)
    return a, b


compile_inputs = make_inputs(TEST_LENGTHS[0])
torch._dynamo.mark_dynamic(compile_inputs[0], 0, min=1, max=16384)
torch._dynamo.mark_dynamic(compile_inputs[1], 0, min=1, max=16384)

compile_fn = torch.compile(eager, backend="inductor", dynamic=True)

for length in TEST_LENGTHS:
    inputs = compile_inputs if length == TEST_LENGTHS[0] else make_inputs(length)
    eager_ret = eager(*inputs)
    compiled_ret = compile_fn(*inputs)
    torch.testing.assert_close(eager_ret, compiled_ret)
    print(f"validated backed-symbol add with length={length}")
