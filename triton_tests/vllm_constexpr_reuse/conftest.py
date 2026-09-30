import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import pytest

from observe import Recorder


def pytest_addoption(parser):
    group = parser.getgroup("constexpr reuse")
    group.addoption("--npu-device", default="npu:0")
    group.addoption("--require-npu", action="store_true", help="Fail instead of skip without NPU")
    group.addoption("--verify-reuse", action="store_true", help="Require D/T binary reuse")
    group.addoption("--reuse-report", type=Path)
    group.addoption("--reuse-collect", type=Path, help="Write selected node IDs without importing NPU dependencies")


def pytest_configure(config):
    config.addinivalue_line("markers", "npu: requires torch_npu and Triton-Ascend")
    config._reuse_records = {}
    config._reuse_outcomes = {}
    config._reuse_versions = {}
    if config.getoption("--verify-reuse") and os.environ.get("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE") != "1":
        raise pytest.UsageError("--verify-reuse requires TRITON_ASCEND_ENABLE_DYNAMIC_REUSE=1 before process start")


def pytest_collection_finish(session):
    config = session.config
    path = config.getoption("--reuse-collect")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([item.nodeid for item in session.items], indent=2) + "\n")
    if config.getoption("--verify-reuse") and len(session.items) != 1:
        raise pytest.UsageError("strict reuse needs one scenario per process; use run_suite.py --strict-reuse")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    item.config._reuse_outcomes.setdefault(item.nodeid, {})[report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    path = config.getoption("--reuse-report")
    if path:
        for record in config._reuse_records.values():
            record["distinct_selected_binaries"] = len({item["hash"] for item in record["launches"]})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": 1,
            "mode": os.environ.get("TRITON_ASCEND_ENABLE_DYNAMIC_REUSE", "0"),
            "strict": config.getoption("--verify-reuse"),
            "exitstatus": int(exitstatus),
            "python": platform.python_version(),
            "source_revision": json.loads(Path(__file__).with_name("sources.json").read_text())["revision"],
            "versions": config._reuse_versions,
            "cases": config._reuse_records,
            "outcomes": config._reuse_outcomes,
        }, indent=2) + "\n")


@pytest.fixture(scope="session")
def npu(request):
    missing = [name for name in ("torch", "torch_npu", "triton") if importlib.util.find_spec(name) is None]
    if missing:
        message = "NPU prerequisites missing: " + ", ".join(missing)
        if request.config.getoption("--require-npu"):
            pytest.fail(message)
        pytest.skip(message)
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    triton = importlib.import_module("triton")
    if not torch.npu.is_available():
        if request.config.getoption("--require-npu"):
            pytest.fail("torch.npu.is_available() is false")
        pytest.skip("NPU unavailable")
    device = request.config.getoption("--npu-device")
    torch.npu.set_device(device)
    request.config._reuse_versions.update({
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "triton": triton.__version__, "device": device,
        "device_name": torch.npu.get_device_name(),
    })
    return torch, device


@pytest.fixture
def case(npu, request):
    torch, device = npu
    torch.manual_seed(20260930)
    record = {"launches": [], "outputs": {}, "reuse_assertions": 0}
    request.config._reuse_records[request.node.nodeid] = record
    return Recorder(torch, device, request.config.getoption("--verify-reuse"), record)


def load_module(request, filename):
    # Steps inside a scenario share these objects. The formal runner also uses
    # one process per scenario because source-keyed registries can cross objects.
    # Stable names across off/on subprocesses, no sys.modules patching of vLLM.
    name = "reuse_case_" + hashlib.sha256((filename + request.node.nodeid).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    originals = {
        name: (fn.src, tuple(fn.arg_names), tuple(fn.constexprs))
        for name, fn in vars(module).items() if hasattr(fn, "constexprs")
    }
    yield module
    for name, before in originals.items():
        fn = getattr(module, name)
        assert (fn.src, tuple(fn.arg_names), tuple(fn.constexprs)) == before, f"original JIT mutated: {name}"
    # Keep module references valid for any background compilation until process exit.


@pytest.fixture
def kernels(npu, request):
    yield from load_module(request, "kernels.py")


@pytest.fixture
def probes(npu, request):
    yield from load_module(request, "probes.py")
