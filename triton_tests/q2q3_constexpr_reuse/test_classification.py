import ast
import copy
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

from analysis_adapter import Pointer, SourceJIT, analyzer
from cases import CASES, SCHEDULE, STATIC, UNKNOWN
from run_analysis import analyze_case, bind_arguments, find_root


class KnownClassificationGap(AssertionError):
    """Only the exact documented mismatch may become an expected failure."""


def parameters():
    for case in CASES:
        for name in case.expected:
            marks = []
            if name in case.gaps:
                marks.append(pytest.mark.xfail(strict=True, raises=KnownClassificationGap,
                                               reason=case.gaps[name].issue))
            yield pytest.param(case, name, id=f"{case.id}.{name}", marks=marks)


@pytest.mark.parametrize("case,name", parameters())
def test_constexpr_classification(case, name, classification_records):
    record = classification_records[case.id]
    row = next(row for row in record["parameters"] if row["parameter"] == name)
    message = (f"{record['path']}:{record['definition_line']} {case.kernel}.{name}: "
               f"expected={row['expected']}, actual={row['actual']}; "
               f"oracle={row['why']}; reasons={row['reasons']}")
    if row["known_gap"]:
        raise KnownClassificationGap(message)
    assert row["matches"], message
    if row["actual"] in (STATIC, UNKNOWN):
        assert row["reasons"], "StaticRequired/Unknown must explain the decision"
    assert all(reason["jit_relative_line"] >= 0 for reason in row["reasons"])


def renamed(fn, arguments):
    tree = fn.parse()
    local_names = set(fn.arg_names) | {n.id for n in ast.walk(tree)
                                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    names = {name: f"renamed_{i}" for i, name in enumerate(sorted(local_names))}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

        def visit_arg(self, node):
            node.arg = names.get(node.arg, node.arg)
            return node

    node = Rename().visit(tree)
    node.body[0].name = "renamed_kernel"
    clone = copy.copy(fn)
    clone.src = ast.unparse(node)
    clone.arg_names = [names[name] for name in fn.arg_names]
    clone.params = [SimpleNamespace(**{**vars(p), "name": names[p.name]}) for p in fn.params]
    clone.signature = inspect.Signature([p.replace(name=names[p.name]) for p in fn.signature.parameters.values()])
    return clone, {names[name]: value for name, value in arguments.items()}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_classification_does_not_depend_on_names(case, classification_roots):
    triton_root, kernel_root = classification_roots
    with analyzer(triton_root) as api:
        fn = api.loader.kernel(kernel_root / case.collection / "src/kernels" / case.path, case.kernel)
        arguments = bind_arguments(api, fn, case.arguments)
        clone, renamed_arguments = renamed(fn, arguments)
        before = api.analyze(fn, arguments, fn.cache_key)
        after = api.analyze(clone, renamed_arguments, "renamed-source")

    def decisions(profile):
        return [(d.position, d.classification.value, sorted({reason for reason, _ in d.reasons}))
                for d in profile.decisions]

    # ast.unparse changes physical lines; compare reasons, not line numbers.
    assert decisions(before) == decisions(after)
    assert before.plan.dynamic == after.plan.dynamic
    assert before.recipe == after.recipe


@pytest.mark.parametrize("mode", ["q3_jagged_to_dense", "q3_jagged_to_dense_no_fusion"])
def test_both_local_jit_helpers_are_analyzed(mode, classification_roots):
    triton_root, kernel_root = classification_roots
    case = next(c for c in CASES if c.id == mode)
    with analyzer(triton_root) as api:
        fn = api.loader.kernel(kernel_root / case.collection / "src/kernels" / case.path, case.kernel)
        profile = api.analyze(fn, bind_arguments(api, fn, case.arguments), fn.cache_key)
        assert {child.fn.name for _, child in profile.plan.helpers} == {
            "tensor_elementwise_add", "tensor_elementwise_mul"}


@pytest.mark.parametrize("arch,mode,expected", [
    ("Ascend950PR_9579", "simd_simt_template", SCHEDULE),
    ("Ascend950PR_9579", "simd", UNKNOWN),
    ("Ascend910B", "simd", UNKNOWN),
])
def test_existing_vllm_schedule_control(classification_roots, arch, mode, expected):
    """A real existing positive control; Q2/Q3 tiles have no matching T rule."""
    triton_root, _ = classification_roots
    toolkits = find_root(None, "TESTTOOLKITS_ROOT", "TestToolkits")
    path = toolkits / "triton_tests/vllm_constexpr_reuse/kernels.py"
    with analyzer(triton_root) as api:
        fn = api.loader.kernel(path, "muls_add_kernel")
        arguments = dict(zip(fn.arg_names, (Pointer("float32"), Pointer("float32"), Pointer("float32"),
                                            0.5, 5000, 5, 1024)))
        allowed = api.config.supports_tile_reuse(SimpleNamespace(backend="npu", arch=arch),
                                               SimpleNamespace(compile_mode=mode))
        profile = api.analyze(fn, arguments, fn.cache_key, allow_tile_reuse=allowed)
        assert [(fn.arg_names[d.position], d.classification.value) for d in profile.decisions] == [("BLOCK_SIZE", expected)]
        assert bool(profile.recipe) == (expected == SCHEDULE)
        if profile.recipe:
            assert profile.recipe.rule == "MaskedGridStrideElementwiseV1"


def test_exhausted_analysis_is_conservative(classification_roots):
    triton_root, kernel_root = classification_roots
    case = next(c for c in CASES if c.id == "q2_store_lowrank_int32")
    with analyzer(triton_root) as api:
        api.config.ANALYSIS_NODE_BUDGET = 0
        result = analyze_case(api, kernel_root, case)
        assert not result["dynamic"]
        assert result["recipe"] is None
        assert {row["actual"] for row in result["parameters"]} == {UNKNOWN}
        assert all("ANALYSIS_BUDGET_EXHAUSTED" in {r["code"] for r in row["reasons"]}
                   for row in result["parameters"])


def test_source_loading_never_executes_imports_or_decorators(classification_roots, tmp_path):
    triton_root, kernel_root = classification_roots
    original = kernel_root / "Q2TritonKernel/src/kernels/store_lowrank.py"
    poisoned = tmp_path / "poisoned.py"
    poisoned.write_text("raise AssertionError('module was executed')\n" + original.read_text(), encoding="utf-8")
    with analyzer(triton_root) as api:
        fn = api.loader.kernel(poisoned, "_store_label_cache_triton_kernel")
        assert isinstance(fn, SourceJIT)
        assert fn.src == api.loader.kernel(original, fn.name).src
        case = next(c for c in CASES if c.id == "q2_store_lowrank")
        assert api.analyze(fn, bind_arguments(api, fn, case.arguments), fn.cache_key).decisions
        with pytest.raises(AssertionError, match="must not execute"):
            fn()
        with pytest.raises(AssertionError, match="must not read"):
            Pointer("float32").data_ptr()


def test_adapter_restores_modules_after_failure(classification_roots, monkeypatch):
    sentinel = ModuleType("triton")
    monkeypatch.setitem(sys.modules, "triton", sentinel)
    observed = {name: value for name, value in sys.modules.items()
                if name == "triton" or name.startswith(("triton.", "torch", "_q2q3_ascend_reuse"))}
    with pytest.raises(RuntimeError, match="deliberate"):
        with analyzer(classification_roots[0]):
            raise RuntimeError("deliberate")
    assert observed == {name: value for name, value in sys.modules.items()
                        if name == "triton" or name.startswith(("triton.", "torch", "_q2q3_ascend_reuse"))}
