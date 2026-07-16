
import os
bs = int(os.getenv('BATCH_SIZE', 128))

import torch

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.append(repo_root)
from TestToolkits.inductor_tests.bench_utils import profile, resolve_device

TARGET_DEVICE = resolve_device()
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.allow_buffer_reuse = False
if TARGET_DEVICE.type == "npu":
    import torch_npu._inductor.fx_passes.post_custom_passes

    torch._inductor.config.post_grad_custom_post_pass = (
        torch_npu._inductor.fx_passes.post_custom_passes.run_register_post_custom_passes
    )
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.generate_intermediate_hooks = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.1+cpu
# torch cuda version: None
# torch git version: e2d141dbde55c2a4370fac5165b0561b6af4798b


# torch.cuda.is_available()==False, no GPU info collected

# triton_unk_fused_full_where & triton_unk_fused_mul_sum

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        full_default = torch.ops.aten.full.default([bs, 38], 0, dtype = torch.int64, layout = torch.strided, device = TARGET_DEVICE, pin_memory = False)
        where = torch.ops.aten.where.self(arg1_1, full_default, arg0_1);  arg1_1 = full_default = arg0_1 = None
        embedding = torch.ops.aten.embedding.default(arg2_1, where);  arg2_1 = where = None
        mul = torch.ops.aten.mul.Tensor(embedding, arg3_1);  embedding = arg3_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul, [1]);  mul = None
        unsqueeze = torch.ops.aten.unsqueeze.default(sum_1, 1);  sum_1 = None
        return (unsqueeze,)
        
def load_args(reader):
    buf0 = reader.storage(None, bs*38*8, device=TARGET_DEVICE, dtype_hint=torch.int64)
    reader.tensor(buf0, (bs, 38), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, bs*38, device=TARGET_DEVICE, dtype_hint=torch.bool)
    reader.tensor(buf1, (bs, 38), dtype=torch.bool, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 20969728, device=TARGET_DEVICE)
    reader.tensor(buf2, (81913, 64), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, bs*38*4, device=TARGET_DEVICE)
    reader.tensor(buf3, (bs, 38, 1), is_leaf=True)  # arg3_1

load_args._version = 0
mod = Repro()

if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        # To run it separately, do 
        mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)

        # performance
        import json
        eager_fn = lambda: mod(*args)
        eager_res = profile(eager_fn, device=TARGET_DEVICE)
        eager_perf_dump_file = os.path.join(
            "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{bs}-eager-perf.json"
        )
        with open(eager_perf_dump_file, 'w', encoding='utf-8') as f:
            json.dump(eager_res, f, ensure_ascii=False, indent=4)

        mod = torch.compile(mod, dynamic=False)
        mod(*args)

        fn = lambda: mod(*args)
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
