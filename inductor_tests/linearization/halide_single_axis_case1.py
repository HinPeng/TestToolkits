"""CPU Halide repro for the single-axis permute and reshape layout path."""

import argparse

import torch
from torch._inductor.utils import run_and_get_code


BATCH = 2
SEQ_LEN = 3
NUM_HEADS = 8
HEAD_DIM = 16
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM


def single_axis_layout(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Keep the add/permute/reshape/add chain from SingleAxisCase1."""
    y = a + b
    y = y.permute(2, 0, 1, 3)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2] * y.shape[3])
    return c + y


def mark_dynamic_axes(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    for tensor in (a, b):
        torch._dynamo.mark_dynamic(tensor, 0, min=1, max=8)
        torch._dynamo.mark_dynamic(tensor, 2, min=1, max=16)
    torch._dynamo.mark_dynamic(c, 0, min=1, max=16)
    torch._dynamo.mark_dynamic(c, 1, min=1, max=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("static", "marked_dynamic"),
        default="marked_dynamic",
        help="Compile with static shapes or mark batch and sequence axes dynamic",
    )
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    a = torch.randn(args.batch, NUM_HEADS, args.seq_len, HEAD_DIM)
    b = torch.randn_like(a)
    c = torch.randn(args.seq_len, args.batch, HIDDEN_SIZE)
    if args.mode == "marked_dynamic":
        mark_dynamic_axes(a, b, c)

    eager = single_axis_layout(a, b, c)
    assert eager.shape == (args.seq_len, args.batch, HIDDEN_SIZE)

    compiled = torch.compile(
        single_axis_layout,
        backend="inductor",
        dynamic=None if args.mode == "marked_dynamic" else False,
        fullgraph=True,
        options={
            "cpu_backend": "halide",
            "halide.scheduler_cpu": "Adams2019",
        },
    )
    actual, source_blocks = run_and_get_code(compiled, a, b, c)
    torch.testing.assert_close(actual, eager)

    source = "\n\n".join(source_blocks)
    if "@hl.generator" not in source or "hl.Var(" not in source:
        raise RuntimeError("Inductor did not emit Halide Generator source")

    print("# mode", args.mode)
    print("# output shape", tuple(actual.shape))
    print("# halide kernel dsl")
    print(source)


if __name__ == "__main__":
    main()
