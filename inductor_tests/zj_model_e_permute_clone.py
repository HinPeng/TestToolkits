import torch
import torch_npu


def eager(args):
    arg0_1 = args[0]
    view = torch.ops.aten.view.default(arg0_1, [-1, 80, 80, 8]);  arg0_1 = None
    permute = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
    # permute += 1
    clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format); del permute
    return (clone,)


if __name__ == "__main__":
    shape = [381, 80, 640]
    arg0_1 = torch.randn(shape, dtype=torch.float16, device='npu')

    eager_output, = eager((arg0_1,))
    torch.npu.synchronize()

    compile_fn = torch.compile(eager, dynamic=False)
    compile_output, = compile_fn((arg0_1,))
    torch.npu.synchronize()

    torch.testing.assert_close(eager_output, compile_output)
