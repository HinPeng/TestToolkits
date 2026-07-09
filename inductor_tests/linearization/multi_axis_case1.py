import argparse
import json
import time
import unittest
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


DEFAULT_BS = 128
DEFAULT_SEQ_LEN = 300
HIDDEN_SIZE = 384
HEAD_DIM = 128
ADDMM_IN_FEATURES = 128
VIEW6_WIDTH = 16
SEQ_EXPANSION = HEAD_DIM // VIEW6_WIDTH
COMPILE_DYNAMIC = {
    "static": False,
    "marked_dynamic": None,
    "fully_symbolic": True,
}
DEFAULT_DEVICE = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"


class MultiAxisCase1(torch.nn.Module):
    def forward(self, arg4_1, view_3, permute):
        bs = view_3.shape[0]
        seq_len = view_3.shape[1]
        view_3_flat = torch.ops.aten.view.default(view_3, [bs * seq_len, ADDMM_IN_FEATURES])
        view_3 = None
        addmm = torch.ops.aten.addmm.default(arg4_1, view_3_flat, permute)
        arg4_1 = view_3_flat = permute = None
        view_4 = torch.ops.aten.view.default(addmm, [bs, seq_len, HIDDEN_SIZE])
        addmm = None
        view_5 = torch.ops.aten.view.default(view_4, [bs, seq_len, 3, HEAD_DIM])
        view_4 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_5, 0)
        view_5 = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4])
        unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(permute_1, -2)
        permute_1 = None
        clone = torch.ops.aten.clone.default(squeeze, memory_format=torch.contiguous_format)
        squeeze = None
        select = torch.ops.aten.select.int(clone, 0, 0)
        view_6 = torch.ops.aten.view.default(select, [bs, seq_len * SEQ_EXPANSION, VIEW6_WIDTH])
        select = None
        permute_2 = torch.ops.aten.permute.default(view_6, [1, 0, 2])
        view_6 = None
        mul_1 = torch.ops.aten.mul.Tensor(permute_2, 0.25)
        permute_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(mul_1, 0)
        mul_1 = None
        return (unsqueeze_1,)


def make_inputs(device=DEFAULT_DEVICE, dtype=None, seed=0, bs=DEFAULT_BS, seq_len=DEFAULT_SEQ_LEN):
    if dtype is None:
        dtype = torch.float16 if device == "npu" else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    arg4_1 = torch.randn((HIDDEN_SIZE,), dtype=dtype, generator=generator).to(device)
    view_3 = torch.randn((bs, seq_len, ADDMM_IN_FEATURES), dtype=dtype, generator=generator).to(device)
    permute = torch.randn((ADDMM_IN_FEATURES, HIDDEN_SIZE), dtype=dtype, generator=generator).to(device)
    return arg4_1, view_3, permute


def mark_dynamic_inputs(inputs, min_bs=1, max_bs=None, min_seq_len=1, max_seq_len=None):
    bs, seq_len = inputs[1].shape[:2]
    max_bs = max(DEFAULT_BS, int(bs)) if max_bs is None else max_bs
    max_seq_len = max(DEFAULT_SEQ_LEN, int(seq_len)) if max_seq_len is None else max_seq_len
    torch._dynamo.mark_dynamic(inputs[1], 0, min=min_bs, max=max_bs)
    torch._dynamo.mark_dynamic(inputs[1], 1, min=min_seq_len, max=max_seq_len)
    return inputs


def synchronize(device):
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def benchmark_callable(fn, warmup=10, iters=50, device=DEFAULT_DEVICE):
    for _ in range(max(warmup, 0)):
        fn()
    synchronize(device)

    started = time.perf_counter()
    for _ in range(max(iters, 1)):
        fn()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "avg_ms": elapsed_ms / max(iters, 1),
        "iters": max(iters, 1),
        "warmup": max(warmup, 0),
    }


def compile_model(model, device=DEFAULT_DEVICE, backend="auto", mode="static"):
    if mode not in COMPILE_DYNAMIC:
        raise ValueError(f"Unsupported compile mode: {mode}")
    selected_backend = backend if backend != "auto" else ("aot_eager" if str(device) == "cpu" else "inductor")
    dynamic = COMPILE_DYNAMIC[mode]
    return torch.compile(model, backend=selected_backend, dynamic=dynamic), selected_backend, dynamic


def assert_outputs_close(eager_out, compiled_out):
    dtype = eager_out.dtype
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-5, 1e-5
    torch.testing.assert_close(eager_out, compiled_out, rtol=rtol, atol=atol)


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


def run_case(
    device=DEFAULT_DEVICE,
    dtype=None,
    warmup=10,
    iters=50,
    backend="auto",
    mode="static",
    bs=DEFAULT_BS,
    seq_len=DEFAULT_SEQ_LEN,
):
    model = MultiAxisCase1().eval()
    inputs = make_inputs(device=device, dtype=dtype, bs=bs, seq_len=seq_len)
    compile_inputs = tuple(tensor.clone() for tensor in inputs)
    if mode == "marked_dynamic":
        mark_dynamic_inputs(compile_inputs)

    with torch.no_grad():
        eager_out, = model(*inputs)
        synchronize(device)

        compiled_model, selected_backend, dynamic = compile_model(
            MultiAxisCase1().eval(), device=device, backend=backend, mode=mode
        )
        compiled_out, = compiled_model(*compile_inputs)
        synchronize(device)
        assert_outputs_close(eager_out, compiled_out)

        eager_perf = benchmark_callable(lambda: model(*inputs), warmup=warmup, iters=iters, device=device)

        # Exclude compilation overhead from the benchmark window.
        compiled_model(*compile_inputs)
        synchronize(device)
        compiled_perf = benchmark_callable(
            lambda: compiled_model(*compile_inputs), warmup=warmup, iters=iters, device=device
        )

    speedup = eager_perf["avg_ms"] / compiled_perf["avg_ms"] if compiled_perf["avg_ms"] else 0.0
    summary = {
        "device": device,
        "dtype": str(inputs[0].dtype).replace("torch.", ""),
        "mode": mode,
        "bs": bs,
        "seq_len": seq_len,
        "compile_backend": selected_backend,
        "compile_dynamic": dynamic,
        "output_shape": list(eager_out.shape),
        "eager_avg_ms": eager_perf["avg_ms"],
        "compiled_avg_ms": compiled_perf["avg_ms"],
        "speedup": speedup,
    }
    return summary, eager_perf, compiled_perf


class MultiAxisCase1Test(unittest.TestCase):
    def test_make_inputs_respects_bs_and_seq_len(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128)

        self.assertEqual(inputs[0].shape, (384,))
        self.assertEqual(inputs[1].shape, (64, 128, 128))
        self.assertEqual(inputs[2].shape, (128, 384))

        eager_out, = MultiAxisCase1().eval()(*inputs)
        self.assertEqual(eager_out.shape, (1, 1024, 64, 16))

    def test_static_compile_match(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=128, seq_len=300)

        eager_mod = MultiAxisCase1().eval()
        compile_mod, backend, dynamic = compile_model(MultiAxisCase1().eval(), device="cpu", mode="static")

        eager_out, = eager_mod(*inputs)
        compile_out, = compile_mod(*inputs)

        self.assertEqual(backend, "aot_eager")
        self.assertFalse(dynamic)
        self.assertEqual(eager_out.shape, (1, 2400, 128, 16))
        torch.testing.assert_close(eager_out, compile_out, rtol=1e-5, atol=1e-5)

    def test_marked_dynamic_compile_matches_two_shapes(self):
        compile_mod, backend, dynamic = compile_model(
            MultiAxisCase1().eval(), device="cpu", mode="marked_dynamic"
        )
        inputs1 = tuple(tensor.clone() for tensor in make_inputs(device="cpu", dtype=torch.float32, bs=128, seq_len=300))
        inputs2 = tuple(tensor.clone() for tensor in make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128))
        mark_dynamic_inputs(inputs1)
        mark_dynamic_inputs(inputs2)

        eager_out1, = MultiAxisCase1().eval()(*(tensor.clone() for tensor in inputs1))
        eager_out2, = MultiAxisCase1().eval()(*(tensor.clone() for tensor in inputs2))
        compile_out1, = compile_mod(*inputs1)
        compile_out2, = compile_mod(*inputs2)

        self.assertEqual(backend, "aot_eager")
        self.assertIsNone(dynamic)
        self.assertEqual(compile_out1.shape, (1, 2400, 128, 16))
        self.assertEqual(compile_out2.shape, (1, 1024, 64, 16))
        torch.testing.assert_close(eager_out1, compile_out1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out2, compile_out2, rtol=1e-5, atol=1e-5)

    def test_fully_symbolic_compile_matches_two_shapes(self):
        compile_mod, backend, dynamic = compile_model(
            MultiAxisCase1().eval(), device="cpu", mode="fully_symbolic"
        )
        inputs1 = make_inputs(device="cpu", dtype=torch.float32, bs=128, seq_len=300)
        inputs2 = make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128)

        eager_out1, = MultiAxisCase1().eval()(*(tensor.clone() for tensor in inputs1))
        eager_out2, = MultiAxisCase1().eval()(*(tensor.clone() for tensor in inputs2))
        compile_out1, = compile_mod(*inputs1)
        compile_out2, = compile_mod(*inputs2)

        self.assertEqual(backend, "aot_eager")
        self.assertTrue(dynamic)
        self.assertEqual(compile_out1.shape, (1, 2400, 128, 16))
        self.assertEqual(compile_out2.shape, (1, 1024, 64, 16))
        torch.testing.assert_close(eager_out1, compile_out1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out2, compile_out2, rtol=1e-5, atol=1e-5)

    def test_benchmark_callable_reports_positive_duration(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128)
        mod = MultiAxisCase1().eval()

        result = benchmark_callable(lambda: mod(*inputs), warmup=1, iters=2)

        self.assertIn("avg_ms", result)
        self.assertGreater(result["avg_ms"], 0.0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu / cuda / npu")
    parser.add_argument("--dtype", default=None, help="float16 / float32 / bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--backend", default="auto", help="auto / inductor / aot_eager / eager")
    parser.add_argument(
        "--mode",
        default="static",
        choices=("static", "marked_dynamic", "fully_symbolic"),
        help="Compilation mode",
    )
    parser.add_argument("--bs", type=int, default=DEFAULT_BS, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length")
    parser.add_argument("--run-tests", action="store_true", help="Run embedded regression tests and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MultiAxisCase1Test)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)

    summary, eager_perf, compiled_perf = run_case(
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        backend=args.backend,
        mode=args.mode,
        bs=args.bs,
        seq_len=args.seq_len,
    )

    base_dir = Path(__file__).parent / "log"
    stem = Path(__file__).stem
    dump_json(base_dir / f"{stem}-eager-perf.json", eager_perf)
    dump_json(base_dir / f"{stem}-perf.json", compiled_perf)
    dump_json(base_dir / f"{stem}-summary.json", summary)

    print(
        f"device={summary['device']} dtype={summary['dtype']} mode={summary['mode']} "
        f"backend={summary['compile_backend']} "
        f"output_shape={tuple(summary['output_shape'])} "
        f"eager={summary['eager_avg_ms']:.3f} ms "
        f"compile={summary['compiled_avg_ms']:.3f} ms "
        f"speedup={summary['speedup']:.3f}x"
    )


if __name__ == "__main__":
    main()
