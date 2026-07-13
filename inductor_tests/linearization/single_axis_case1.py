import argparse
import json
import sys
import unittest
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from TestToolkits.inductor_tests.bench_utils import profile, synchronize


DEFAULT_BS = 128
DEFAULT_SEQ_LEN = 300
NUM_HEADS = 8
HEAD_DIM = 16
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM
COMPILE_DYNAMIC = {
    "static": False,
    "marked_dynamic": None,
    "fully_symbolic": True,
}
DEFAULT_DEVICE = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"


class SingleAxisCase1(torch.nn.Module):
    def forward(self, a, b, c):
        y = torch.ops.aten.add.Tensor(a, b)
        a = b = None
        y = torch.ops.aten.permute.default(y, [2, 0, 1, 3])
        bs = y.shape[1]
        seq_len = y.shape[0]
        head_dim = y.shape[3]
        hidden_size = y.shape[2] * head_dim
        y = torch.ops.aten.reshape.default(y, [seq_len, bs, hidden_size])
        y = torch.ops.aten.add.Tensor(c, y)
        return y


def make_inputs(
    device=DEFAULT_DEVICE,
    dtype=None,
    seed=0,
    bs=DEFAULT_BS,
    seq_len=DEFAULT_SEQ_LEN,
    head_dim=HEAD_DIM,
):
    if dtype is None:
        dtype = torch.float16 if device == "npu" else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    hidden_size = NUM_HEADS * head_dim
    a = torch.randn((bs, NUM_HEADS, seq_len, head_dim), dtype=dtype, generator=generator).to(device)
    b = torch.randn((bs, NUM_HEADS, seq_len, head_dim), dtype=dtype, generator=generator).to(device)
    c = torch.randn((seq_len, bs, hidden_size), dtype=dtype, generator=generator).to(device)
    return a, b, c


def mark_dynamic_inputs(
    inputs,
    min_bs=1,
    max_bs=None,
    min_seq_len=1,
    max_seq_len=None,
    min_head_dim=1,
    max_head_dim=None,
):
    bs = int(inputs[0].shape[0])
    seq_len = int(inputs[0].shape[2])
    head_dim = int(inputs[0].shape[3])
    max_bs = max(DEFAULT_BS, bs) if max_bs is None else max_bs
    max_seq_len = max(DEFAULT_SEQ_LEN, seq_len) if max_seq_len is None else max_seq_len
    max_head_dim = max(HEAD_DIM, head_dim) if max_head_dim is None else max_head_dim
    for tensor in inputs[:2]:
        torch._dynamo.mark_dynamic(tensor, 0, min=min_bs, max=max_bs)
        torch._dynamo.mark_dynamic(tensor, 2, min=min_seq_len, max=max_seq_len)
        torch._dynamo.mark_dynamic(tensor, 3, min=min_head_dim, max=max_head_dim)
    torch._dynamo.mark_dynamic(inputs[2], 0, min=min_seq_len, max=max_seq_len)
    torch._dynamo.mark_dynamic(inputs[2], 1, min=min_bs, max=max_bs)
    torch._dynamo.mark_dynamic(
        inputs[2],
        2,
        min=NUM_HEADS * min_head_dim,
        max=NUM_HEADS * max_head_dim,
    )
    return inputs


def compile_model(model, device=DEFAULT_DEVICE, backend="auto", mode="static"):
    if mode not in COMPILE_DYNAMIC:
        raise ValueError(f"Unsupported compile mode: {mode}")
    selected_backend = backend if backend != "auto" else ("aot_eager" if str(device) == "cpu" else "inductor")
    dynamic = COMPILE_DYNAMIC[mode]
    try:
        compiled_model = torch.compile(model, backend=selected_backend, dynamic=dynamic)
    except ModuleNotFoundError as exc:
        if str(device) != "cpu" or selected_backend != "aot_eager":
            raise
        print(f"[single_axis_case1] Falling back to eager module on CPU: {exc}")
        selected_backend = "eager_fallback"
        compiled_model = model
    return compiled_model, selected_backend, dynamic


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
    head_dim=HEAD_DIM,
):
    model = SingleAxisCase1().eval()
    inputs = make_inputs(device=device, dtype=dtype, bs=bs, seq_len=seq_len, head_dim=head_dim)
    compile_inputs = tuple(tensor.clone() for tensor in inputs)
    if mode == "marked_dynamic":
        mark_dynamic_inputs(compile_inputs)

    with torch.no_grad():
        eager_out = model(*inputs)
        synchronize(device)

        compiled_model, selected_backend, dynamic = compile_model(
            SingleAxisCase1().eval(), device=device, backend=backend, mode=mode
        )
        compiled_out = compiled_model(*compile_inputs)
        synchronize(device)
        assert_outputs_close(eager_out, compiled_out)

        eager_perf = profile(lambda: model(*inputs), warmup=warmup, active=iters, device=device)

        compiled_model(*compile_inputs)
        synchronize(device)
        compiled_perf = profile(
            lambda: compiled_model(*compile_inputs), warmup=warmup, active=iters, device=device
        )

    eager_total_us = eager_perf.get("__total_us__", 0.0)
    compiled_total_us = compiled_perf.get("__total_us__", 0.0)
    speedup = eager_total_us / compiled_total_us if compiled_total_us else 0.0
    summary = {
        "device": device,
        "dtype": str(inputs[0].dtype).replace("torch.", ""),
        "mode": mode,
        "bs": bs,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "eager_bench_backend": eager_perf.get("__backend__", "unknown"),
        "compiled_bench_backend": compiled_perf.get("__backend__", "unknown"),
        "compile_backend": selected_backend,
        "compile_dynamic": dynamic,
        "output_shape": list(eager_out.shape),
        "eager_total_us": eager_total_us,
        "compiled_total_us": compiled_total_us,
        "speedup": speedup,
    }
    return summary, eager_perf, compiled_perf


class SingleAxisCase1Test(unittest.TestCase):
    def test_make_inputs_respects_bs_and_seq_len(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128)

        self.assertEqual(inputs[0].shape, (64, 8, 128, 16))
        self.assertEqual(inputs[1].shape, (64, 8, 128, 16))
        self.assertEqual(inputs[2].shape, (128, 64, 128))

        eager_out = SingleAxisCase1().eval()(*inputs)
        self.assertEqual(eager_out.shape, (128, 64, 128))

    def test_make_inputs_respects_head_dim(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128, head_dim=32)

        self.assertEqual(inputs[0].shape, (64, 8, 128, 32))
        self.assertEqual(inputs[1].shape, (64, 8, 128, 32))
        self.assertEqual(inputs[2].shape, (128, 64, 256))

        eager_out = SingleAxisCase1().eval()(*inputs)
        self.assertEqual(eager_out.shape, (128, 64, 256))

    def test_static_compile_match(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=128, seq_len=300)

        eager_mod = SingleAxisCase1().eval()
        compile_mod, backend, dynamic = compile_model(SingleAxisCase1().eval(), device="cpu", mode="static")

        eager_out = eager_mod(*inputs)
        compile_out = compile_mod(*inputs)

        self.assertIn(backend, ("aot_eager", "eager_fallback"))
        self.assertFalse(dynamic)
        self.assertEqual(eager_out.shape, (300, 128, 128))
        torch.testing.assert_close(eager_out, compile_out, rtol=1e-5, atol=1e-5)

    def test_marked_dynamic_compile_matches_two_shapes(self):
        compile_mod, backend, dynamic = compile_model(
            SingleAxisCase1().eval(), device="cpu", mode="marked_dynamic"
        )
        inputs1 = tuple(tensor.clone() for tensor in make_inputs(device="cpu", dtype=torch.float32, bs=128, seq_len=300))
        inputs2 = tuple(tensor.clone() for tensor in make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128))
        mark_dynamic_inputs(inputs1)
        mark_dynamic_inputs(inputs2)

        eager_out1 = SingleAxisCase1().eval()(*(tensor.clone() for tensor in inputs1))
        eager_out2 = SingleAxisCase1().eval()(*(tensor.clone() for tensor in inputs2))
        compile_out1 = compile_mod(*inputs1)
        compile_out2 = compile_mod(*inputs2)

        self.assertIn(backend, ("aot_eager", "eager_fallback"))
        self.assertIsNone(dynamic)
        self.assertEqual(compile_out1.shape, (300, 128, 128))
        self.assertEqual(compile_out2.shape, (128, 64, 128))
        torch.testing.assert_close(eager_out1, compile_out1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out2, compile_out2, rtol=1e-5, atol=1e-5)

    def test_marked_dynamic_compile_matches_symbolic_head_dim(self):
        compile_mod, backend, dynamic = compile_model(
            SingleAxisCase1().eval(), device="cpu", mode="marked_dynamic"
        )
        inputs1 = tuple(
            tensor.clone()
            for tensor in make_inputs(
                device="cpu", dtype=torch.float32, bs=128, seq_len=300, head_dim=16
            )
        )
        inputs2 = tuple(
            tensor.clone()
            for tensor in make_inputs(
                device="cpu", dtype=torch.float32, bs=128, seq_len=300, head_dim=32
            )
        )
        mark_dynamic_inputs(inputs1)
        mark_dynamic_inputs(inputs2)

        eager_out1 = SingleAxisCase1().eval()(*(tensor.clone() for tensor in inputs1))
        eager_out2 = SingleAxisCase1().eval()(*(tensor.clone() for tensor in inputs2))
        compile_out1 = compile_mod(*inputs1)
        compile_out2 = compile_mod(*inputs2)

        self.assertIn(backend, ("aot_eager", "eager_fallback"))
        self.assertIsNone(dynamic)
        self.assertEqual(compile_out1.shape, (300, 128, 128))
        self.assertEqual(compile_out2.shape, (300, 128, 256))
        torch.testing.assert_close(eager_out1, compile_out1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out2, compile_out2, rtol=1e-5, atol=1e-5)

    def test_fully_symbolic_compile_matches_two_shapes(self):
        compile_mod, backend, dynamic = compile_model(
            SingleAxisCase1().eval(), device="cpu", mode="fully_symbolic"
        )
        inputs1 = make_inputs(device="cpu", dtype=torch.float32, bs=128, seq_len=300, head_dim=16)
        inputs2 = make_inputs(device="cpu", dtype=torch.float32, bs=64, seq_len=128, head_dim=32)

        eager_out1 = SingleAxisCase1().eval()(*(tensor.clone() for tensor in inputs1))
        eager_out2 = SingleAxisCase1().eval()(*(tensor.clone() for tensor in inputs2))
        compile_out1 = compile_mod(*inputs1)
        compile_out2 = compile_mod(*inputs2)

        self.assertIn(backend, ("aot_eager", "eager_fallback"))
        self.assertTrue(dynamic)
        self.assertEqual(compile_out1.shape, (300, 128, 128))
        self.assertEqual(compile_out2.shape, (128, 64, 256))
        torch.testing.assert_close(eager_out1, compile_out1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out2, compile_out2, rtol=1e-5, atol=1e-5)

    def test_run_case_reports_positive_total_us(self):
        _, eager_perf, compiled_perf = run_case(
            device="cpu",
            dtype=torch.float32,
            warmup=1,
            iters=2,
            mode="static",
            bs=64,
            seq_len=128,
        )

        self.assertIn("__total_us__", eager_perf)
        self.assertIn("__total_us__", compiled_perf)
        self.assertGreater(eager_perf["__total_us__"], 0.0)
        self.assertGreater(compiled_perf["__total_us__"], 0.0)


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
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM, help="Per-head hidden size")
    parser.add_argument("--run-tests", action="store_true", help="Run embedded regression tests and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SingleAxisCase1Test)
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
        head_dim=args.head_dim,
    )

    base_dir = Path(__file__).parent / "log"
    stem = Path(__file__).stem
    dump_json(base_dir / f"{stem}-eager-perf.json", eager_perf)
    dump_json(base_dir / f"{stem}-perf.json", compiled_perf)
    dump_json(base_dir / f"{stem}-summary.json", summary)

    print(
        f"device={summary['device']} dtype={summary['dtype']} mode={summary['mode']} "
        f"head_dim={summary['head_dim']} "
        f"backend={summary['compile_backend']} "
        f"output_shape={tuple(summary['output_shape'])} "
        f"eager={summary['eager_total_us']:.3f} us "
        f"compile={summary['compiled_total_us']:.3f} us "
        f"speedup={summary['speedup']:.3f}x"
    )


if __name__ == "__main__":
    main()
