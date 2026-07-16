
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
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.1+cpu
# torch cuda version: None
# torch git version: e2d141dbde55c2a4370fac5165b0561b6af4798b


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 300]);  arg0_1 = None
        view_1 = torch.ops.aten.view.default(arg1_1, [-1, 300]);  arg1_1 = None
        embedding = torch.ops.aten.embedding.default(arg2_1, view);  arg2_1 = view = None
        view_2 = torch.ops.aten.view.default(view_1, [-1, 300, 1]);  view_1 = None
        mul = torch.ops.aten.mul.Tensor(embedding, view_2);  embedding = view_2 = None
        view_3 = torch.ops.aten.view.default(mul, [38400, 128]);  mul = None
        permute = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm = torch.ops.aten.addmm.default(arg3_1, view_3, permute);  arg3_1 = view_3 = permute = None
        view_4 = torch.ops.aten.view.default(addmm, [128, 300, 384]);  addmm = None
        view_5 = torch.ops.aten.view.default(view_4, [128, 300, 3, 128]);  view_4 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_5, 0);  view_5 = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(permute_1, -2);  permute_1 = None
        clone = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone, 0, 0)
        select_1 = torch.ops.aten.select.int(clone, 0, 1)
        view_6 = torch.ops.aten.view.default(select, [128, 2400, 16]);  select = None
        permute_2 = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(select_1, [128, 2400, 16]);  select_1 = None
        permute_3 = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        mul_1 = torch.ops.aten.mul.Tensor(permute_2, 0.25);  permute_2 = None
        permute_5 = torch.ops.aten.permute.default(permute_3, [0, 2, 1]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(mul_1, permute_5);  mul_1 = permute_5 = None
        amax = torch.ops.aten.amax.default(bmm, [-1], True)
        sub = torch.ops.aten.sub.Tensor(bmm, amax);  bmm = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        return (div,)
        
def load_args(reader):
    buf0 = reader.storage(None, 307200, device=TARGET_DEVICE, dtype_hint=torch.int64)
    reader.tensor(buf0, (128, 300), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 153600, device=TARGET_DEVICE)
    reader.tensor(buf1, (128, 300), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 5120000, device=TARGET_DEVICE)
    reader.tensor(buf2, (10000, 128), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1536, device=TARGET_DEVICE)
    reader.tensor(buf3, (384,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 196608, device=TARGET_DEVICE)
    reader.tensor(buf4, (384, 128), is_leaf=True)  # arg4_1
    
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
