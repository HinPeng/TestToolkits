from pathlib import Path

import pytest

from analysis_adapter import analyzer
from cases import CASES
from run_analysis import analyze_case, find_kernel_root, find_root
from snapshot import verify


def pytest_addoption(parser):
    group = parser.getgroup("Q2/Q3 constexpr classification")
    group.addoption("--triton-ascend-root", type=Path, help="Checkout containing third_party/ascend/backend/reuse")
    group.addoption("--kernel-root", type=Path, help="Optional upstream TritonAscendTest root; defaults to bundled kernels.py")


@pytest.fixture(scope="session")
def classification_roots(request):
    try:
        return (
            find_root(request.config.getoption("--triton-ascend-root"), "TRITON_ASCEND_ROOT", "triton-ascend"),
            find_kernel_root(request.config.getoption("--kernel-root")),
        )
    except FileNotFoundError as error:
        raise pytest.UsageError(str(error)) from error


@pytest.fixture(scope="session")
def classification_records(classification_roots):
    triton_root, kernel_root = classification_roots
    if kernel_root is None:
        verify()
    # Do not leave adapted modules installed across other test suites.
    with analyzer(triton_root) as api:
        return {case.id: analyze_case(api, kernel_root, case) for case in CASES}
