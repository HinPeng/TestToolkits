import torch
import triton
import triton.language as tl

torch._inductor.config.force_disable_caches = True
torch._dynamo.config.recompile_limit = 4096

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.fake_tensor_cache_enabled = False

current_device = "npu"


def eager(mask, where_info):
    mask = torch.clamp(mask, min=0, max=1)
    zmask = torch.reshape(mask, [-1]) > 0
    z_indices = torch.nonzero(zmask).squeeze(-1)
    length = z_indices.shape[0]

    full_1: "i64[u0, 512]" = torch.ops.aten.full.default(
        [length, 512],
        64,
        dtype=torch.int64,
        layout=torch.strided,
        device=current_device,
        pin_memory=False,
    )

    size_full = torch.ops.aten.sym_size.int(full_1, 0)
    full_2: "i64[u0, 512]" = torch.ops.aten.full.default(
        [size_full, 512],
        128,
        dtype=torch.int64,
        layout=torch.strided,
        device=current_device,
        pin_memory=False,
    )

    where_t = torch.ops.aten.where.self(where_info, full_2, full_1)
    return where_t


a1 = torch.cat(
    (
        torch.zeros((128,), device=current_device),
        torch.ones((512,), device=current_device),
        torch.zeros((128,), device=current_device),
    )
)
b1 = (torch.arange(512, device=current_device) % 2 == 0).reshape(1, 512)

eager_ret = eager(a1, b1)

compile_fn = torch.compile(eager, dynamic=False)
compiled_ret = compile_fn(a1, b1)

torch.testing.assert_close(eager_ret, compiled_ret)
