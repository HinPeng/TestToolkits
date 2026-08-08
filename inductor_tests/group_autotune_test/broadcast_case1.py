"""broadcast_case: 分组开启 / 分组关闭 / 静态编译 / eager 对比.

NPU (A5):
  - eager: 不编译, 直接执行
  - static: dynamic=False, per-batch
  - dyn_nogroup: dynamic=None, 分组关闭
  - dyn_group: dynamic=None, 分组开启 (仅NPU)

GPU (L20):
  - eager: 不编译, 直接执行
  - static: dynamic=False, per-batch
  - dyn_nogroup: dynamic=None
"""

import argparse
import gc
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


DEFAULT_BATCHES = [64, 78, 102, 134, 156, 178, 200, 220, 245, 256]
DEFAULT_N = 2048
DEFAULT_M = 16
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50

COMPILE_DYNAMIC = {
    "static": False,
    "dyn_nogroup": None,
    "dyn_group": None,
}

NPU_MODES = ["eager", "static", "dyn_nogroup", "dyn_group"]
GPU_MODES = ["eager", "static", "dyn_nogroup"]

DEFAULT_DEVICE = (
    "npu"
    if hasattr(torch, "npu") and torch.npu.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def set_grouping(enabled):
    """配置 NPU 符号化形状分组自动调优."""
    if torch_npu is None:
        return
    import torch_npu._inductor.config as npu_config

    npu_config.enable_symbolic_shape_group_autotune = enabled
    if enabled:
        npu_config.symbolic_group_allow_templates = ("pointwise",)
    torch._dynamo.reset()


def clear_memory(device=DEFAULT_DEVICE):
    """释放设备显存."""
    gc.collect()
    if device == "npu":
        torch.npu.empty_cache()
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class BroadcastCase1(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def make_inputs(device=DEFAULT_DEVICE, dtype=None, seed=0, bs=DEFAULT_BATCHES[0], n=DEFAULT_N, m=DEFAULT_M):
    if dtype is None:
        dtype = torch.float16 if device == "npu" else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x = torch.randn((bs, n, m), dtype=dtype, generator=generator).to(device)
    y = torch.randn((m,), dtype=dtype, generator=generator).to(device)
    return x, y


def mark_dynamic_inputs(inputs, min_bs=1, max_bs=None):
    bs = int(inputs[0].shape[0])
    max_bs = max(DEFAULT_BATCHES[-1], bs) if max_bs is None else max_bs
    torch._dynamo.mark_dynamic(inputs[0], 0, min=min_bs, max=max_bs)
    return inputs


def compile_model(model, device=DEFAULT_DEVICE, backend="auto", mode="static"):
    if mode not in COMPILE_DYNAMIC and mode != "eager":
        raise ValueError(f"Unsupported compile mode: {mode}")
    if mode == "eager":
        return model, "eager", None
    selected_backend = backend if backend != "auto" else ("aot_eager" if str(device) == "cpu" else "inductor")
    dynamic = COMPILE_DYNAMIC[mode]
    if mode == "dyn_group":
        set_grouping(True)
    elif mode == "dyn_nogroup":
        set_grouping(False)
    try:
        compiled_model = torch.compile(model, backend=selected_backend, dynamic=dynamic)
    except ModuleNotFoundError as exc:
        if str(device) != "cpu" or selected_backend != "aot_eager":
            raise
        print(f"[broadcast_case1] Falling back to eager module on CPU: {exc}")
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


def available_modes(device=DEFAULT_DEVICE):
    if device == "npu":
        return list(NPU_MODES)
    return list(GPU_MODES)


def run_case(
    device=DEFAULT_DEVICE,
    dtype=None,
    warmup=DEFAULT_WARMUP,
    iters=DEFAULT_ITERS,
    backend="auto",
    mode="static",
    batches=None,
    n=DEFAULT_N,
    m=DEFAULT_M,
):
    if batches is None:
        batches = list(DEFAULT_BATCHES)

    model = BroadcastCase1().eval()
    details = []

    # dyn_* 模式: 用第一个 batch 编译一次, 后续 batch 复用
    shared_compiled = None
    selected_backend = "eager"
    dynamic = None

    if mode in ("dyn_nogroup", "dyn_group"):
        warmup_inputs = make_inputs(device=device, dtype=dtype, bs=batches[0], n=n, m=m)
        mark_dynamic_inputs(warmup_inputs)
        shared_compiled, selected_backend, dynamic = compile_model(
            BroadcastCase1().eval(), device=device, backend=backend, mode=mode
        )
        with torch.no_grad():
            shared_compiled(*warmup_inputs)
            synchronize(device)

    for bs in batches:
        inputs = make_inputs(device=device, dtype=dtype, bs=bs, n=n, m=m)

        with torch.no_grad():
            eager_out = model(*inputs)
            synchronize(device)

            if mode == "eager":
                compiled_out = eager_out
                mode_perf = profile(lambda: model(*inputs), warmup=warmup, active=iters, device=device)
            elif mode == "static":
                # static: 每个 batch 单独编译
                torch._dynamo.reset()
                compiled_model, selected_backend, dynamic = compile_model(
                    BroadcastCase1().eval(), device=device, backend=backend, mode=mode
                )
                compile_inputs = tuple(t.clone() for t in inputs)
                compiled_out = compiled_model(*compile_inputs)
                synchronize(device)
                assert_outputs_close(eager_out, compiled_out)
                compiled_model(*compile_inputs)
                synchronize(device)
                mode_perf = profile(
                    lambda: compiled_model(*compile_inputs), warmup=warmup, active=iters, device=device
                )
            else:
                # dyn_nogroup / dyn_group: 复用 shared_compiled
                compile_inputs = tuple(t.clone() for t in inputs)
                compiled_out = shared_compiled(*compile_inputs)
                synchronize(device)
                assert_outputs_close(eager_out, compiled_out)
                mode_perf = profile(
                    lambda: shared_compiled(*compile_inputs), warmup=warmup, active=iters, device=device
                )

            eager_perf = profile(lambda: model(*inputs), warmup=warmup, active=iters, device=device)

        eager_total_us = eager_perf.get("__total_us__", 0.0)
        mode_total_us = mode_perf.get("__total_us__", 0.0)
        speedup = eager_total_us / mode_total_us if mode_total_us else 0.0
        detail = {
            "bs": bs,
            "shape": [bs, n, m],
            "dtype": str(inputs[0].dtype).replace("torch.", ""),
            "mode": mode,
            "eager_bench_backend": eager_perf.get("__backend__", "unknown"),
            "mode_bench_backend": mode_perf.get("__backend__", "unknown"),
            "compile_backend": selected_backend,
            "compile_dynamic": dynamic,
            "output_shape": list(eager_out.shape),
            "eager_total_us": round(eager_total_us, 1),
            "mode_total_us": round(mode_total_us, 1),
            "speedup": round(speedup, 3),
        }
        details.append(detail)
        clear_memory(device)
        if mode == "static":
            torch._dynamo.reset()

    total_eager_us = round(sum(d["eager_total_us"] for d in details), 1)
    total_mode_us = round(sum(d["mode_total_us"] for d in details), 1)
    total_speedup = round(total_eager_us / total_mode_us, 3) if total_mode_us else 0.0
    summary = {
        "device": device,
        "mode": mode,
        "batches": batches,
        "n": n,
        "m": m,
        "details": details,
        "total_eager_us": total_eager_us,
        "total_mode_us": total_mode_us,
        "total_speedup": total_speedup,
    }
    return summary


def run_comparison(
    device=DEFAULT_DEVICE,
    dtype=None,
    warmup=DEFAULT_WARMUP,
    iters=DEFAULT_ITERS,
    backend="auto",
    modes=None,
    batches=None,
    n=DEFAULT_N,
    m=DEFAULT_M,
):
    if modes is None:
        modes = available_modes(device)
    if batches is None:
        batches = list(DEFAULT_BATCHES)

    summaries = []
    for mode in modes:
        print(f"\n--- mode={mode} ---")
        summary = run_case(
            device=device,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            backend=backend,
            mode=mode,
            batches=batches,
            n=n,
            m=m,
        )
        summaries.append(summary)

    # 对比表
    print(f"\n{'=' * 70}")
    print(f"broadcast_case 对比: device={device}, batches={batches}, n={n}, m={m}")
    print(f"{'=' * 70}")
    header = f"  {'bs':>7} | " + " | ".join(f"{mode:>14}" for mode in modes)
    print(header)
    print(f"  {'-' * 7}-+-" + "-+-".join(["-" * 14] * len(modes)))

    mode_details = {s["mode"]: {d["bs"]: d["mode_total_us"] for d in s["details"]} for s in summaries}
    for bs in batches:
        vals = [f"{mode_details[m].get(bs, 0):>14.1f}" for m in modes]
        print(f"  {bs:>7} | " + " | ".join(vals))

    totals = [f"{s['total_mode_us']:>14.1f}" for s in summaries]
    print(f"  {'TOTAL':>7} | " + " | ".join(totals))

    return summaries


class BroadcastCase1Test(unittest.TestCase):
    def test_make_inputs_respects_bs(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=64)
        self.assertEqual(inputs[0].shape, (64, DEFAULT_N, DEFAULT_M))
        self.assertEqual(inputs[1].shape, (DEFAULT_M,))

    def test_make_inputs_respects_n_and_m(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=64, n=128, m=32)
        self.assertEqual(inputs[0].shape, (64, 128, 32))
        self.assertEqual(inputs[1].shape, (32,))

    def test_eager_output_shape(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=128)
        eager_out = BroadcastCase1().eval()(*inputs)
        self.assertEqual(eager_out.shape, (128, DEFAULT_N, DEFAULT_M))

    def test_static_compile_match(self):
        inputs = make_inputs(device="cpu", dtype=torch.float32, bs=128)
        eager_mod = BroadcastCase1().eval()
        compile_mod, backend, dynamic = compile_model(BroadcastCase1().eval(), device="cpu", mode="static")

        eager_out = eager_mod(*inputs)
        compile_out = compile_mod(*inputs)

        self.assertIn(backend, ("aot_eager", "eager_fallback"))
        self.assertFalse(dynamic)
        torch.testing.assert_close(eager_out, compile_out, rtol=1e-5, atol=1e-5)

    def test_dyn_nogroup_compile_matches_two_shapes(self):
        compile_mod, backend, dynamic = compile_model(
            BroadcastCase1().eval(), device="cpu", mode="dyn_nogroup"
        )
        inputs1 = tuple(t.clone() for t in make_inputs(device="cpu", dtype=torch.float32, bs=128))
        inputs2 = tuple(t.clone() for t in make_inputs(device="cpu", dtype=torch.float32, bs=64))
        mark_dynamic_inputs(inputs1)
        mark_dynamic_inputs(inputs2)

        eager_out1 = BroadcastCase1().eval()(*(t.clone() for t in inputs1))
        eager_out2 = BroadcastCase1().eval()(*(t.clone() for t in inputs2))
        compile_out1 = compile_mod(*inputs1)
        compile_out2 = compile_mod(*inputs2)

        self.assertIn(backend, ("aot_eager", "eager_fallback"))
        self.assertIsNone(dynamic)
        torch.testing.assert_close(eager_out1, compile_out1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(eager_out2, compile_out2, rtol=1e-5, atol=1e-5)

    def test_run_case_reports_positive_total_us(self):
        summary = run_case(
            device="cpu",
            dtype=torch.float32,
            warmup=1,
            iters=2,
            mode="static",
            batches=[64, 128],
        )
        self.assertEqual(len(summary["details"]), 2)
        for d in summary["details"]:
            self.assertGreater(d["eager_total_us"], 0.0)
            self.assertGreater(d["mode_total_us"], 0.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu / cuda / npu")
    parser.add_argument("--dtype", default=None, help="float16 / float32 / bfloat16")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--backend", default="auto", help="auto / inductor / aot_eager / eager")
    parser.add_argument(
        "--mode",
        default=None,
        choices=("eager", "static", "dyn_nogroup", "dyn_group"),
        help="单模式测试; 省略则运行该设备全部模式对比",
    )
    parser.add_argument("--batches", default=None, help="逗号分隔的 batch 列表")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="第二维大小")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="第三维大小")
    parser.add_argument("--run-tests", action="store_true", help="运行内嵌回归测试后退出")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BroadcastCase1Test)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)

    batches = [int(b) for b in args.batches.split(",")] if args.batches else list(DEFAULT_BATCHES)

    base_dir = Path(__file__).parent / "log"
    stem = Path(__file__).stem

    if args.mode is not None:
        summary = run_case(
            device=args.device,
            dtype=args.dtype,
            warmup=args.warmup,
            iters=args.iters,
            backend=args.backend,
            mode=args.mode,
            batches=batches,
            n=args.n,
            m=args.m,
        )
        dump_json(base_dir / f"{stem}-{args.mode}-summary.json", summary)
        print(
            f"device={summary['device']} mode={summary['mode']} "
            f"batches={summary['batches']} "
            f"total_eager={summary['total_eager_us']:.3f} us "
            f"total_mode={summary['total_mode_us']:.3f} us "
            f"speedup={summary['total_speedup']:.3f}x"
        )
    else:
        summaries = run_comparison(
            device=args.device,
            dtype=args.dtype,
            warmup=args.warmup,
            iters=args.iters,
            backend=args.backend,
            modes=None,
            batches=batches,
            n=args.n,
            m=args.m,
        )
        dump_json(base_dir / f"{stem}-comparison.json", summaries)


if __name__ == "__main__":
    main()
