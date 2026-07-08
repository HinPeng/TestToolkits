import os
bs = int(os.getenv('BATCH_SIZE', 128))
numel = bs * 16

from easydict import EasyDict as edict

import torch
import torch.nn as nn

from torch._dynamo.testing import rand_strided


class SoftmaxAddEtc(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        
    def forward(self, inputs):
        view_135 = inputs[0]
        
        # File: /usr/local/python3.11.13/lib/python3.11/site-packages/transformers/modeling_utils.py:1672 in invert_attention_mask, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        full_default_3: "i64[1, 1, 1, 2048]" = torch.ops.aten.full.default([1, 1, 1, numel], 1, dtype = torch.int64, layout = torch.strided, device = self.configs.device, pin_memory = False)

        # File: /usr/local/python3.11.13/lib/python3.11/site-packages/transformers/modeling_utils.py:1678 in invert_attention_mask, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        _npu_dtype_cast_4: "f32[1, 1, 1, 2048]" = torch.ops.npu._npu_dtype_cast.default(full_default_3, torch.float32);  full_default_3 = None

        # File: /usr/local/python3.11.13/lib/python3.11/site-packages/transformers/modeling_utils.py:1679 in invert_attention_mask, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
        sub_7: "f32[1, 1, 1, 2048]" = torch.ops.aten.sub.Tensor(1.0, _npu_dtype_cast_4);  _npu_dtype_cast_4 = None
        mul_31: "f32[1, 1, 1, 2048]" = torch.ops.aten.mul.Tensor(sub_7, -3.4028234663852886e+38);  sub_7 = None

        # File: /data/z00605466/onerec-jd/recsys-examples/examples/onerec/model/t5_layer.py:392 in forward, code: position_bias = torch.zeros(
        full_default_6: "f32[1, 16, 2048, 2048]" = torch.ops.aten.full.default([1, 16, 2048, numel], 0, dtype = torch.float32, layout = torch.strided, device = self.configs.device, pin_memory = False)

        # File: /data/z00605466/onerec-jd/recsys-examples/examples/onerec/model/t5_layer.py:411 in forward, code: position_bias + mask
        add_32: "f32[1, 16, 2048, 2048]" = torch.ops.aten.add.Tensor(full_default_6, mul_31);  full_default_6 = mul_31 = None

        # File: /data/z00605466/onerec-jd/recsys-examples/examples/onerec/model/t5_layer.py:421 in forward, code: scores += position_bias_masked
        add_33: "f32[1, 16, 2048, 2048]" = torch.ops.aten.add.Tensor(view_135, add_32);  view_135 = None

        # File: /data/z00605466/onerec-jd/recsys-examples/examples/onerec/model/t5_layer.py:422 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        amax_5: "f32[1, 16, 2048, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_10: "f32[1, 16, 2048, 2048]" = torch.ops.aten.sub.Tensor(add_33, amax_5);  add_33 = amax_5 = None
        exp_5: "f32[1, 16, 2048, 2048]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_6: "f32[1, 16, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_9: "f32[1, 16, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        return div_9
        

def get_input_data(configs):
    view_135 = rand_strided((1, 16, 2048, numel), (16*2048*numel, 2048*numel, numel, 1), device=configs.device, dtype=torch.float32)
    return (view_135,)


if __name__ == "__main__":
    configs = edict({
        "model": "SoftmaxAddEtc",
        "is_compile_mode": True,
        "device": "npu",
    })
    
    inputs = get_input_data(configs)
    mod = SoftmaxAddEtc(configs)

    # performance
    import json
    from bench_utils import profile

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
