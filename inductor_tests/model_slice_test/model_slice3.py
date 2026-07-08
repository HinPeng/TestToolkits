
import os
bs = int(os.getenv('BATCH_SIZE', 128))

import torch
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
torch._inductor.config.generate_intermediate_hooks = True
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

# # triton_unk_fused__npu_dtype_cast_binary_cross_

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        minimum = torch.ops.aten.minimum.default(full_default, arg0_1);  full_default = None
        abs_1 = torch.ops.aten.abs.default(arg0_1);  arg0_1 = None
        neg = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        exp = torch.ops.aten.exp.default(neg);  neg = None
        log1p = torch.ops.aten.log1p.default(exp);  exp = None
        sub = torch.ops.aten.sub.Tensor(minimum, log1p);  minimum = log1p = None
        sub_1 = torch.ops.aten.sub.Tensor(arg1_1, sub);  arg1_1 = sub = None
        mean = torch.ops.aten.mean.default(sub_1);  sub_1 = None
        return (mean,)
        
def load_args(reader):
    numel = bs * 50 + 63
    buf0 = reader.storage(None, numel*4, device=device(type='npu', index=0))
    reader.tensor(buf0, (numel,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, numel*4, device=device(type='npu', index=0))
    reader.tensor(buf1, (numel,), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()

if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        # To run it separately, do 
        mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)

        # performance
        import json
        from bench_utils import profile

        eager_fn = lambda: mod(*args)
        eager_res = profile(eager_fn)
        eager_perf_dump_file = os.path.join(
            "./log", os.path.splitext(os.path.basename(__file__))[0] + f"-{bs}-eager-perf.json"
        )
        with open(eager_perf_dump_file, 'w', encoding='utf-8') as f:
            json.dump(eager_res, f, ensure_ascii=False, indent=4)

        mod = torch.compile(mod, dynamic=False)
        mod(*args)

        fn = lambda: mod(*args)
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
