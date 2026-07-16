import importlib.util
import os
import sys
import types
import unittest

from pathlib import Path
from unittest import mock


BENCH_UTILS_PATH = Path(__file__).with_name("bench_utils.py")


def load_bench_utils(npu_available=False, cuda_available=False):
    fake_torch = types.ModuleType("torch")

    class Device:
        def __init__(self, spec):
            self.type, _, index = str(spec).partition(":")
            self.index = int(index) if index else None

        def __str__(self):
            return self.type if self.index is None else f"{self.type}:{self.index}"

    fake_torch.device = Device
    fake_torch.npu = types.SimpleNamespace(
        is_available=lambda: npu_available, synchronize=lambda: None
    )
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available, synchronize=lambda: None
    )
    spec = importlib.util.spec_from_file_location("bench_utils_under_test", BENCH_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module


class BenchUtilsTest(unittest.TestCase):
    def _resolve_device_and_timer(self, bench_utils):
        self.assertTrue(
            hasattr(bench_utils, "resolve_device"),
            "bench_utils must expose resolve_device",
        )
        self.assertTrue(
            hasattr(bench_utils, "resolve_timer"),
            "bench_utils must expose resolve_timer",
        )
        return bench_utils.resolve_device, bench_utils.resolve_timer

    def test_timer_defaults_and_overrides(self):
        npu = load_bench_utils(npu_available=True)
        resolve_device, resolve_timer = self._resolve_device_and_timer(npu)
        with mock.patch.dict(os.environ, {}, clear=True):
            npu_device = resolve_device()
        self.assertEqual(resolve_timer(npu_device), "profiler")
        self.assertEqual(resolve_timer(npu_device, "event"), "event")

        cuda = load_bench_utils(cuda_available=True)
        resolve_device, resolve_timer = self._resolve_device_and_timer(cuda)
        with mock.patch.dict(os.environ, {"DEVICE": "cuda:1"}, clear=True):
            cuda_device = resolve_device()
        self.assertEqual(str(cuda_device), "cuda:1")
        self.assertEqual(resolve_timer(cuda_device), "event")
        with self.assertRaisesRegex(ValueError, "NPU-only"):
            resolve_timer(cuda_device, "profiler")

    def test_invalid_configuration_is_explicit(self):
        bench_utils = load_bench_utils()
        resolve_device, resolve_timer = self._resolve_device_and_timer(bench_utils)
        with self.assertRaisesRegex(RuntimeError, "No supported accelerator"):
            resolve_device()
        with mock.patch.dict(os.environ, {"DEVICE": "cuda"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Requested CUDA device is unavailable"):
                resolve_device()
        with self.assertRaisesRegex(ValueError, "profiler.*event"):
            resolve_timer(types.SimpleNamespace(type="npu"), "clock")

    def test_event_timer_uses_do_bench_mean_and_converts_to_us(self):
        bench_utils = load_bench_utils(cuda_available=True)
        triton = types.ModuleType("triton")
        testing = types.ModuleType("triton.testing")
        testing.do_bench = mock.Mock(return_value=2.5)
        triton.testing = testing
        with mock.patch.dict(sys.modules, {"triton": triton, "triton.testing": testing}):
            result = bench_utils.profile(lambda: None, warmup=7, active=11, device="cuda")
        testing.do_bench.assert_called_once_with(mock.ANY, warmup=7, rep=11, return_mode="mean")
        self.assertEqual(result, {"__total_us__": 2500.0, "__backend__": "event"})


if __name__ == "__main__":
    unittest.main()
