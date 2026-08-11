"""Small CUDA vector cases used to collect PyTorch 2.13.0 Inductor evidence.

The cases intentionally keep the semantic shape/layout pressure from the
walkthrough plan while avoiding model-specific dependencies.  The runner is
responsible for compilation, correctness checks, and artifact capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


TensorFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    source_reference: str
    trace_shape: tuple[int, ...]
    canonical_shape: tuple[int, ...]
    fn: TensorFn


def v0_pointwise(x: torch.Tensor) -> torch.Tensor:
    """Three fused pointwise expressions on one contiguous 1-D input."""

    return torch.relu(x + 1) * torch.sigmoid(x)


def t2_transpose_clone(x: torch.Tensor) -> torch.Tensor:
    """Rectangular view/permute/unary/clone pressure case.

    The reshape is legal without a copy for both the trace and canonical
    shapes.  The final clone makes the output allocation/store visible while
    preserving the non-contiguous access pattern in the producer.
    """

    viewed = x.view(-1, 80, 40, 8)
    transposed = viewed.permute(0, 2, 1, 3)
    return torch.sin(transposed).clone()


CASES: dict[str, CaseSpec] = {
    "V0": CaseSpec(
        case_id="V0",
        title="contiguous pointwise chain",
        source_reference="vector plan V0; upstream pointwise baseline",
        trace_shape=(1024,),
        canonical_shape=(8192,),
        fn=v0_pointwise,
    ),
    "T2": CaseSpec(
        case_id="T2",
        title="view -> permute -> sin -> clone",
        source_reference=(
            "pytorch_ascend/test/_inductor/"
            "test_linear_triton_store_semantics.py::"
            "TestLinearTritonStoreSemantics.test_scheduler_semantic_store_path_for_transpose_clone"
        ),
        trace_shape=(7, 80, 320),
        canonical_shape=(381, 80, 320),
        fn=t2_transpose_clone,
    ),
}


def get_case(case_id: str) -> CaseSpec:
    try:
        return CASES[case_id.upper()]
    except KeyError as exc:
        available = ", ".join(sorted(CASES))
        raise ValueError(f"Unknown case {case_id!r}; available cases: {available}") from exc


def make_inputs(
    case_id: str,
    device: torch.device,
    shape_mode: str = "trace",
    dtype: torch.dtype = torch.float32,
) -> tuple[CaseSpec, list[torch.Tensor]]:
    spec = get_case(case_id)
    if shape_mode not in {"trace", "canonical"}:
        raise ValueError("shape_mode must be 'trace' or 'canonical'")

    shape = spec.trace_shape if shape_mode == "trace" else spec.canonical_shape
    if spec.case_id == "V0":
        inputs = [torch.randn(*shape, device=device, dtype=dtype)]
    elif spec.case_id == "T2":
        inputs = [torch.randn(*shape, device=device, dtype=dtype)]
    else:  # pragma: no cover - guarded by CASES
        raise AssertionError(f"No input builder for {spec.case_id}")
    return spec, inputs
