
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
torch._inductor.config.inplace_buffers = True
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

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
        full_default = torch.ops.aten.full.default([], -1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(arg1_1, full_default, arg0_1);  arg1_1 = full_default = arg0_1 = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand = torch.ops.aten.expand.default(div, [128, 4, 50, 50]);  div = None
        view = torch.ops.aten.view.default(expand, [512, 50, 50]);  expand = None
        expand_1 = torch.ops.aten.expand.default(arg2_1, [128, 4, 50, 8]);  arg2_1 = None
        clone = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1 = torch.ops.aten.view.default(clone, [512, 50, 8]);  clone = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2 = torch.ops.aten.view.default(bmm, [128, 4, 50, 8]);  bmm = None
        permute = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_1 = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3 = torch.ops.aten.view.default(clone_1, [128, 50, 32]);  clone_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [6400, 32]);  view_3 = None
        permute_1 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm = torch.ops.aten.addmm.default(arg4_1, view_4, permute_1);  arg4_1 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm, [128, 50, 32]);  addmm = None
        add = torch.ops.aten.add.Tensor(arg5_1, view_5);  arg5_1 = view_5 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        view_6 = torch.ops.aten.view.default(add_2, [6400, 32])
        permute_2 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_6, permute_2);  arg9_1 = view_6 = permute_2 = None
        view_7 = torch.ops.aten.view.default(addmm_1, [128, 50, 128]);  addmm_1 = None
        gt = torch.ops.aten.gt.Scalar(view_7, 0)
        mul_2 = torch.ops.aten.mul.Tensor(view_7, 0.01)
        where_1 = torch.ops.aten.where.self(gt, view_7, mul_2);  gt = view_7 = mul_2 = None
        view_8 = torch.ops.aten.view.default(where_1, [6400, 128]);  where_1 = None
        permute_3 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_8, permute_3);  arg11_1 = view_8 = permute_3 = None
        view_9 = torch.ops.aten.view.default(addmm_2, [128, 50, 32]);  addmm_2 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, view_9);  add_2 = view_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, arg12_1);  mul_3 = arg12_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        return (add_5,)
        
def load_args(reader):
    buf0 = reader.storage(None, 5120000, device=device(type='npu', index=0))
    reader.tensor(buf0, (128, 4, 50, 50), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 128, device=device(type='npu', index=0), dtype_hint=torch.bool)
    reader.tensor(buf1, (128, 1, 1, 1), dtype=torch.bool, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 819200, device=device(type='npu', index=0))
    reader.tensor(buf2, (128, 4, 50, 8), (1600, 8, 32, 1), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 4096, device=device(type='npu', index=0))
    reader.tensor(buf3, (32, 32), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 128, device=device(type='npu', index=0))
    reader.tensor(buf4, (32,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 819200, device=device(type='npu', index=0))
    reader.tensor(buf5, (128, 50, 32), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 128, device=device(type='npu', index=0))
    reader.tensor(buf6, (32,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 128, device=device(type='npu', index=0))
    reader.tensor(buf7, (32,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 16384, device=device(type='npu', index=0))
    reader.tensor(buf8, (128, 32), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='npu', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 16384, device=device(type='npu', index=0))
    reader.tensor(buf10, (32, 128), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 128, device=device(type='npu', index=0))
    reader.tensor(buf11, (32,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 128, device=device(type='npu', index=0))
    reader.tensor(buf12, (32,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 128, device=device(type='npu', index=0))
    reader.tensor(buf13, (32,), is_leaf=True)  # arg13_1

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
