"""Observe ordinary launches; never call a selected binary or adapt its arguments."""

import hashlib
import time


def scalar(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def describe(compiled):
    if compiled is None or not getattr(compiled, "hash", None):
        raise AssertionError("JIT.run must return the actually launched CompiledKernel")
    source = compiled.src
    names = source.fn.arg_names
    constants = {}
    for key, value in source.constants.items():
        if isinstance(key, tuple) and len(key) == 1:
            key = names[key[0]]
        elif isinstance(key, int):
            key = names[key]
        constants[str(key)] = scalar(value)
    return {
        "hash": str(compiled.hash),
        "constants": constants,
        "signature": {str(k): str(v) for k, v in source.signature.items()},
    }


class Recorder:
    def __init__(self, torch, device, strict, record):
        self.torch = torch
        self.device = device
        self.strict = strict
        self.record = record

    def tensor(self, data, dtype=None):
        return self.torch.tensor(data, dtype=dtype, device=self.device)

    def launch(self, kernel, grid, *args, **kwargs):
        start = time.perf_counter()
        compiled = kernel[grid](*args, **kwargs)
        self.torch.npu.synchronize()
        description = describe(compiled)
        description.update({
            "kernel": kernel.__name__,
            "requested": {
                name: scalar(value)
                for name, value in zip(kernel.arg_names, args)
                if not isinstance(value, self.torch.Tensor)
            } | {name: scalar(value) for name, value in kwargs.items()
                 if not isinstance(value, self.torch.Tensor)},
            "grid": "callable" if callable(grid) else list(grid),
            "synchronized_wall_ms": (time.perf_counter() - start) * 1000,
        })
        self.record["launches"].append(description)
        return description

    def check(self, name, actual, expected, *, rtol=0, atol=0):
        actual = actual.detach().cpu().contiguous()
        expected = self.torch.as_tensor(expected, dtype=actual.dtype, device="cpu")
        self.torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        self.record["outputs"][name] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "sha256": hashlib.sha256(actual.view(self.torch.uint8).numpy().tobytes()).hexdigest(),
        }

    def same_binary(self, first, second):
        if self.strict:
            assert first["hash"] == second["hash"], "expected reuse of the warmed binary"
            self.record["reuse_assertions"] += 1

    def dynamic(self, description, *names):
        if self.strict:
            for name in names:
                assert name not in description["constants"], f"{name} still specializes by value"
                assert name in description["signature"], f"{name} missing from runtime ABI"

    @staticmethod
    def static(description, **values):
        for name, value in values.items():
            assert description["constants"].get(name) == value, (
                f"{name} must remain {value}; selected {description['constants']}"
            )
