import os
import time

os.environ.setdefault("INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE", "1")
os.environ.setdefault("INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES", "pointwise")

import torch
import torch_npu  # noqa: F401
from triton.backends.ascend import do_bench_npu


torch._inductor.config.force_disable_caches = True
torch._dynamo.config.recompile_limit = 4096

CURRENT_DEVICE = "npu"
TEST_LENGTHS = (128, 1024, 8192, 16384, 65536, 262144)
MAX_DYNAMIC_LENGTH = max(TEST_LENGTHS)
E2E_WARMUP_ITERS = 20
E2E_ACTIVE_ITERS = 100
KERNEL_WARMUP_ITERS = 5
KERNEL_ACTIVE_ITERS = 30


def eager(a, b):
    return a + b


def make_inputs(length):
    a = torch.randn(length, device=CURRENT_DEVICE)
    b = torch.randn(length, device=CURRENT_DEVICE)
    return a, b


def clone_inputs(inputs):
    return tuple(tensor.clone() for tensor in inputs)


def mark_dynamic_inputs(inputs):
    torch._dynamo.mark_dynamic(inputs[0], 0, min=1, max=MAX_DYNAMIC_LENGTH)
    torch._dynamo.mark_dynamic(inputs[1], 0, min=1, max=MAX_DYNAMIC_LENGTH)
    return inputs


def measure_first_call_ms(fn, inputs):
    torch.npu.synchronize()
    start = time.perf_counter()
    output = fn(*inputs)
    torch.npu.synchronize()
    end = time.perf_counter()
    return output, (end - start) * 1e3


def measure_steady_e2e_ms(fn, inputs, warmup=E2E_WARMUP_ITERS, active=E2E_ACTIVE_ITERS):
    for _ in range(warmup):
        fn(*inputs)
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(active):
        fn(*inputs)
    torch.npu.synchronize()
    end = time.perf_counter()
    return (end - start) * 1e3 / active


def format_ratio(lhs, rhs):
    if rhs == 0:
        return "inf"
    return f"{lhs / rhs:.4f}x"


def benchmark_length(length, symbolic_fn):
    static_fn = torch.compile(eager, backend="inductor", dynamic=False)

    base_inputs = make_inputs(length)
    symbolic_inputs = mark_dynamic_inputs(clone_inputs(base_inputs))
    static_inputs = clone_inputs(base_inputs)
    eager_ret = eager(*base_inputs)

    symbolic_first_output, symbolic_first_call_ms = measure_first_call_ms(symbolic_fn, symbolic_inputs)
    static_first_output, static_first_call_ms = measure_first_call_ms(static_fn, static_inputs)

    torch.testing.assert_close(symbolic_first_output, eager_ret)
    torch.testing.assert_close(static_first_output, eager_ret)
    print(f"validated backed-symbol add with length={length}")

    symbolic_e2e_ms = measure_steady_e2e_ms(symbolic_fn, symbolic_inputs)
    static_e2e_ms = measure_steady_e2e_ms(static_fn, static_inputs)

    # do_bench_npu uses torch_npu.profiler to run kernels repeatedly and returns
    # the average on-device kernel duration in milliseconds.
    symbolic_kernel_ms = do_bench_npu(
        lambda: symbolic_fn(*symbolic_inputs),
        warmup=KERNEL_WARMUP_ITERS,
        active=KERNEL_ACTIVE_ITERS,
    )
    static_kernel_ms = do_bench_npu(
        lambda: static_fn(*static_inputs),
        warmup=KERNEL_WARMUP_ITERS,
        active=KERNEL_ACTIVE_ITERS,
    )

    print(
        "length={length}: "
        "symbolic_first_call_ms={symbolic_first_call_ms:.3f}, "
        "static_first_call_ms={static_first_call_ms:.3f}, "
        "symbolic_e2e_ms={symbolic_e2e_ms:.3f}, "
        "static_e2e_ms={static_e2e_ms:.3f}, "
        "symbolic_kernel_ms={symbolic_kernel_ms:.3f}, "
        "static_kernel_ms={static_kernel_ms:.3f}, "
        "kernel_ratio(symbolic/static)={kernel_ratio}, "
        "e2e_ratio(symbolic/static)={e2e_ratio}".format(
            length=length,
            symbolic_first_call_ms=symbolic_first_call_ms,
            static_first_call_ms=static_first_call_ms,
            symbolic_e2e_ms=symbolic_e2e_ms,
            static_e2e_ms=static_e2e_ms,
            symbolic_kernel_ms=symbolic_kernel_ms,
            static_kernel_ms=static_kernel_ms,
            kernel_ratio=format_ratio(symbolic_kernel_ms, static_kernel_ms),
            e2e_ratio=format_ratio(symbolic_e2e_ms, static_e2e_ms),
        )
    )

    return {
        "length": length,
        "symbolic_first_call_ms": symbolic_first_call_ms,
        "static_first_call_ms": static_first_call_ms,
        "symbolic_e2e_ms": symbolic_e2e_ms,
        "static_e2e_ms": static_e2e_ms,
        "symbolic_kernel_ms": symbolic_kernel_ms,
        "static_kernel_ms": static_kernel_ms,
    }


symbolic_fn = torch.compile(eager, backend="inductor", dynamic=True)
print(
    "kernel_ms is measured by do_bench_npu via torch_npu.profiler, "
    "which repeatedly launches the kernel and averages the on-device execution time."
)

results = []
for length in TEST_LENGTHS:
    results.append(benchmark_length(length, symbolic_fn))

avg_symbolic_kernel_ms = sum(item["symbolic_kernel_ms"] for item in results) / len(results)
avg_static_kernel_ms = sum(item["static_kernel_ms"] for item in results) / len(results)
avg_symbolic_e2e_ms = sum(item["symbolic_e2e_ms"] for item in results) / len(results)
avg_static_e2e_ms = sum(item["static_e2e_ms"] for item in results) / len(results)

print(
    "summary: "
    "avg_symbolic_kernel_ms={avg_symbolic_kernel_ms:.3f}, "
    "avg_static_kernel_ms={avg_static_kernel_ms:.3f}, "
    "avg_kernel_ratio(symbolic/static)={avg_kernel_ratio}, "
    "avg_symbolic_e2e_ms={avg_symbolic_e2e_ms:.3f}, "
    "avg_static_e2e_ms={avg_static_e2e_ms:.3f}, "
    "avg_e2e_ratio(symbolic/static)={avg_e2e_ratio}".format(
        avg_symbolic_kernel_ms=avg_symbolic_kernel_ms,
        avg_static_kernel_ms=avg_static_kernel_ms,
        avg_kernel_ratio=format_ratio(avg_symbolic_kernel_ms, avg_static_kernel_ms),
        avg_symbolic_e2e_ms=avg_symbolic_e2e_ms,
        avg_static_e2e_ms=avg_static_e2e_ms,
        avg_e2e_ratio=format_ratio(avg_symbolic_e2e_ms, avg_static_e2e_ms),
    )
)
