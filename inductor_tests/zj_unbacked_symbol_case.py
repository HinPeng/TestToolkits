import torch
import triton
import triton.language as tl

torch._inductor.config.force_disable_caches = True
torch._dynamo.config.recompile_limit = 4096

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.fake_tensor_cache_enabled = False

current_device = "npu"


def eager(mask, mask2, where_info):
    mask = torch.clamp(mask, min=0, max=1)
    zmask = torch.reshape(mask, [-1]) > 0
    z_indices = torch.nonzero(zmask).squeeze(-1)
    length1 = z_indices.shape[0]

    mask2 = torch.clamp(mask2, min=0, max=1)
    zmask2 = torch.reshape(mask2, [-1]) > 0
    z_indices2 = torch.nonzero(zmask2).squeeze(-1)
    length2 = z_indices2.shape[0]

    full_216: "i64[u0, 512]" = torch.ops.aten.full.default(
        [length1, 512 + length2],
        128,
        dtype=torch.int64,
        layout=torch.strided,
        device=current_device,
        pin_memory=False,
    )

    view1 = torch.ops.aten.view.default(full_216, [256, 2, 512 + length2])
    view2 = torch.ops.aten.view.default(view1, [-1, 2 * (512 + length2)])
    size_full = torch.ops.aten.sym_size.int(view2, 1)
    full_2: "i64[u0, 512]" = torch.ops.aten.full.default(
        [256, size_full],
        128,
        dtype=torch.int64,
        layout=torch.strided,
        device=current_device,
        pin_memory=False,
    )

    where_t = torch.ops.aten.where.default(where_info, full_2, view2)
    return where_t


a1 = torch.cat(
    (
        torch.zeros((128,), device=current_device),
        torch.ones((512,), device=current_device),
        torch.zeros((128,), device=current_device),
    )
)
a2 = torch.cat(
    (
        torch.zeros((128,), device=current_device),
        torch.ones((512,), device=current_device),
        torch.zeros((128,), device=current_device),
    )
)
b1 = torch.ones((256, 1), dtype=torch.bool, device=current_device)

eager_ret = eager(a1, a2, b1)

compile_fn = torch.compile(eager, dynamic=False)
compiled_ret = compile_fn(a1, a2, b1)

torch.testing.assert_close(eager_ret, compiled_ret)
