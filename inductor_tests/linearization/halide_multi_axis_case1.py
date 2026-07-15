"""CPU Halide repro for the 128 -> 8 x 16 multi-axis layout path."""

import argparse

import torch
from torch._inductor.utils import run_and_get_code


BATCH = 2
SEQ_LEN = 3
HEADS = 3
HEAD_DIM = 128
VIEW_WIDTH = 16
EXPANSION = HEAD_DIM // VIEW_WIDTH


def multi_axis_layout(x: torch.Tensor) -> torch.Tensor:
    """Keep the view/permute chain from MultiAxisCase1 without addmm."""
    grouped = x.unsqueeze(0).permute(3, 1, 2, 0, 4).squeeze(-2).contiguous()
    selected = grouped.select(0, 0)
    split_head = selected.view(x.shape[0], x.shape[1] * EXPANSION, VIEW_WIDTH)
    permuted = split_head.permute(1, 0, 2)
    return torch.sin(permuted) + 0.25 * permuted


def mark_dynamic_axes(x: torch.Tensor) -> None:
    torch._dynamo.mark_dynamic(x, 0, min=1, max=8)
    torch._dynamo.mark_dynamic(x, 1, min=1, max=16)


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
    x = torch.randn(args.batch, args.seq_len, HEADS, HEAD_DIM, dtype=torch.float32)
    if args.mode == "marked_dynamic":
        mark_dynamic_axes(x)

    eager = multi_axis_layout(x)
    assert eager.shape == (args.seq_len * EXPANSION, args.batch, VIEW_WIDTH)

    compiled = torch.compile(
        multi_axis_layout,
        backend="inductor",
        dynamic=None if args.mode == "marked_dynamic" else False,
        fullgraph=True,
        options={
            "cpu_backend": "halide",
            "halide.scheduler_cpu": "Adams2019",
        },
    )
    actual, source_blocks = run_and_get_code(compiled, x)
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
