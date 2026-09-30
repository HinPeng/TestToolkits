"""Host checks run even where torch / Triton / NPU are not installed."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from observe import describe
import reference
from run_suite import compare_reports, validate_report
import run_suite
from snapshot import definitions, verify
from catalog import NPU_TEST_FILES, OPERATORS, validate_operator_coverage

ROOT = Path(__file__).resolve().parent


def test_frozen_source_integrity():
    manifest = verify()
    assert len(manifest["functions"]) == 22
    assert manifest["revision"] == "18d0537d919c303952fbe6f3442349d0de42970d"


def test_renamed_probe_preserves_structure():
    names = dict(zip(
        ("muls_add_kernel", "x_ptr", "y_ptr", "output_ptr", "scale", "n_elements", "n_blocks",
         "BLOCK_SIZE", "pid", "num_programs", "block_id", "block_start", "offsets", "mask", "x", "y", "output"),
        ("renamed_pointwise", "a", "b", "result", "factor", "length", "work",
         "CHUNK", "lane", "lanes", "part", "start", "idx", "live", "lhs", "rhs", "value"),
    ))

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

        def visit_arg(self, node):
            node.arg = names.get(node.arg, node.arg)
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            node.name = names.get(node.name, node.name)
            return self.generic_visit(node)

    original = ast.parse(definitions(ROOT / "kernels.py")["muls_add_kernel"][1])
    renamed = ast.parse(definitions(ROOT / "probes.py")["renamed_pointwise"][1])
    assert ast.dump(Rename().visit(original)) == ast.dump(renamed)


def test_slot_reference_with_empty_request_and_padding():
    result = reference.slot_mapping([2, 0, 1], [0, 17, 3], [[[1, 2], [3, 4], [5, 6]]],
                                    [16], [False], 5, -7)
    assert result == [[16, 33, 83, -7, -7]]
    assert reference.slot_mapping([], [], [[]], [16], [False], 3, -9) == [[-9] * 3]
    assert reference.slot_mapping([2], [-1, 17], [[[4]]], [16], [True], 2, -11) == [[-11, 65]]


def test_metadata_reference():
    assert reference.metadata([0, 3, 7], [10, 20], 2, 5, True) == ([0, 1, 3], [10, 18], [7, 16])
    assert reference.metadata([0, 3], [10], 4, 5, False) == ([0, 0], [0], [-777])


def test_bincount_reference_rank_and_padding():
    assert reference.bin_counts([[0, 1, 1, 4, -1], [4, 5, 5, 7, 8]], 4) == [[1, 2, 0, 0], [0] * 4]
    assert reference.bin_counts([[0, 4, 5, 5, 7, 8]], 4, 1) == [[1, 2, 0, 1]]


def test_ngram_min_n_changes_semantics():
    row = [7, 1, 2, 9, 8, 1, 0, 0]
    result = reference.ngram(row, 6, [2, -1], False, 32, 2, 4, 4)
    assert result == ([7, 1, 2, 9, 8, 1, 2, 0], 2, [9, 8, 1, 2], 4, 1)
    assert reference.ngram(row, 6, [2, -1], False, 32, 3, 4, 4)[2:4] == ([-1] * 4, 0)
    assert reference.ngram(row, 6, [2, -1], True, 32, 2, 4, 4) == (row, 1, [-1] * 4, 0, 0)


def test_rejection_modes_are_observably_different():
    target = [[0.4] + [0.6 / 63] * 63] * 4
    args = ([0, 1, 0, 1], target, None, [0.6, 0.8, 0.05, 0.05], [0.9, 0.1], [0, 0])
    normal = reference.rejection(*args, False, False)
    synthetic = reference.rejection(*args, True, False)
    entropy = reference.rejection(*args, False, True)
    assert normal == [[50, -777, -777], [0, 53, -777]]
    assert synthetic == [[0, 51, -777], [0, 1, 61]]
    assert entropy[0] == [0, 51, -777]
    assert entropy != normal


def test_u2_witness_has_missing_tail():
    def covered(block, limit):
        pieces = (300 + block - 1) // block
        return {(index // pieces, col)
                for index in range(limit)
                for col in range((index % pieces) * block, min((index % pieces + 1) * block, 300))}

    wanted = covered(256, 4)
    stale = covered(128, 4)
    assert len(wanted) == 600
    assert wanted - stale == {(1, col) for col in range(128, 300)}


def test_observation_uses_selected_binary_metadata():
    source = SimpleNamespace(fn=SimpleNamespace(arg_names=["x", "PAD_ID", "BLOCK"]),
                             constants={(2,): 128}, signature={"x": "*i32", "PAD_ID": "i32"})
    description = describe(SimpleNamespace(hash="selected", src=source))
    assert description["constants"] == {"BLOCK": 128}
    assert description["signature"]["PAD_ID"] == "i32"
    with pytest.raises(AssertionError, match="actually launched"):
        describe(None)


def report(mode):
    return {"mode": mode, "exitstatus": 0, "strict": False,
            "cases": {"test": {"outputs": {"x": {"sha256": "abc", "shape": [1], "dtype": "int32"}},
                               "launches": [{"hash": "compiled"}], "reuse_assertions": 0}},
            "outcomes": {"test": {"setup": "passed", "call": "passed", "teardown": "passed"}}}


def test_report_comparison_requires_equal_bytes():
    off, on = report("0"), report("1")
    assert compare_reports(off, on) == 1
    on["cases"]["test"]["outputs"]["x"]["sha256"] = "different"
    with pytest.raises(ValueError, match="output bytes differ"):
        compare_reports(off, on)


@pytest.mark.parametrize("failure", ["empty", "no_output", "skipped", "unrecorded", "failed", "strict_without_reuse"])
def test_report_does_not_accept_missing_evidence(failure):
    value = copy.deepcopy(report("1"))
    if failure == "empty":
        value["cases"] = {}
    elif failure == "no_output":
        value["cases"]["test"]["outputs"] = {}
    elif failure == "skipped":
        value["outcomes"]["test"]["call"] = "skipped"
    elif failure == "unrecorded":
        value["outcomes"]["another"] = {"setup": "skipped"}
    elif failure == "failed":
        value["exitstatus"] = 1
    else:
        value["strict"] = True
    with pytest.raises(ValueError):
        validate_report(value)


def test_runner_isolates_each_scenario_and_mode(tmp_path, monkeypatch):
    calls = []
    nodes = ["test_operators.py::first", "test_operators.py::second"]

    def run(command, *, cwd, check, env=None):
        assert check is False
        if "--collect-only" in command:
            path = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--reuse-collect="))
            Path(path).write_text(json.dumps(nodes))
        else:
            calls.append((list(command), dict(env)))
            path = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--reuse-report="))
            node = next(arg for arg in command if arg in nodes)
            evidence = report(env["TRITON_ASCEND_ENABLE_DYNAMIC_REUSE"])
            evidence["strict"] = "--verify-reuse" in command
            value = evidence["cases"].pop("test")
            value["reuse_assertions"] = int(evidence["strict"] and node == nodes[0])
            evidence["cases"][node] = value
            evidence["outcomes"][node] = evidence["outcomes"].pop("test")
            Path(path).write_text(json.dumps(evidence))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_suite.subprocess, "run", run)
    output = tmp_path / "evidence"
    assert run_suite.main(["--strict-reuse", "--select", "first or second", "--output", str(output)]) == 0
    assert [env["TRITON_ASCEND_ENABLE_DYNAMIC_REUSE"] for _, env in calls] == ["0", "0", "1", "1"]
    assert len({env["TRITON_CACHE_DIR"] for _, env in calls}) == 4
    assert all(sum(node in command for node in nodes) == 1 for command, _ in calls)
    assert len(json.loads((output / "on.json").read_text())["cases"]) == 2


@pytest.mark.parametrize("requested", list(OPERATORS))
def test_required_operator_has_source_and_executable_case(requested):
    assert len(OPERATORS) == 19
    _, symbol, test_name = OPERATORS[requested]
    assert symbol in verify()["functions"]
    tests = {node.name: node for file in NPU_TEST_FILES for node in ast.parse((ROOT / file).read_text()).body
             if isinstance(node, ast.FunctionDef)}
    test = tests[test_name]
    assert any(isinstance(node, ast.Attribute) and node.attr == symbol for node in ast.walk(test))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "launch"
               for node in ast.walk(test))


def test_direct_launches_bind_to_original_signatures():
    kernels = {node.name: node for node in ast.parse((ROOT / "kernels.py").read_text()).body
               if isinstance(node, ast.FunctionDef)}
    checked = set()
    for file in NPU_TEST_FILES:
        for call in ast.walk(ast.parse((ROOT / file).read_text())):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "launch" and call.args):
                continue
            entry = call.args[0]
            if not (isinstance(entry, ast.Attribute) and isinstance(entry.value, ast.Name)
                    and entry.value.id == "kernels"):
                continue
            args = kernels[entry.attr].args
            names = [arg.arg for arg in args.args]
            positional = call.args[2:]
            assert len(positional) <= len(names), entry.attr
            bound = set(names[:len(positional)])
            options = {"multibuffer", "force_simt_only"}
            for keyword in call.keywords:
                assert keyword.arg in names or keyword.arg in options, (entry.attr, keyword.arg)
                assert keyword.arg not in bound, (entry.attr, keyword.arg)
                bound.add(keyword.arg)
            required = names[:len(names) - len(args.defaults)] if args.defaults else names
            assert set(required) <= bound, (entry.attr, set(required) - bound)
            checked.add(entry.attr)
    # Slot entries use a locally selected kernel plus **kwargs; all others bind directly.
    expected = {symbol for _, symbol, _ in OPERATORS.values() if "slot_mapping" not in symbol}
    assert expected <= checked


def test_full_run_rejects_frozen_but_unexecuted_operators():
    evidence = report("0")
    evidence["cases"]["test"]["launches"] = [{"kernel": symbol} for _, symbol, _ in OPERATORS.values()]
    assert len(validate_operator_coverage(evidence)) == 19
    evidence["cases"]["test"]["launches"].pop()
    with pytest.raises(ValueError, match="were not executed"):
        validate_operator_coverage(evidence)


def test_prepare_padded_reference():
    assert reference.prepare_padded([0, 2, 3], [1, 1, 4], [0, 1, 4, 8]) == ([0, 1, 7], [0, 2, 0])
    assert reference.prepare_padded([], [], [0]) == ([], [])


def test_greedy_reference_modes_empty_and_disabled():
    args = ([[1, 2], [], [3]], [[1, 5], [], [8]], [20, 21, 22], [0.9, 0.1, 0.1], [0.5, 0.5])
    assert reference.greedy_sample(*args, [True] * 3, False, 2) == [[1, 5, -777], [21, -777, -777], [8, -777, -777]]
    assert reference.greedy_sample(*args, [True, False, True], True, 2) == [[1, -777, -777], [-777] * 3, [3, 22, -777]]


def test_recovered_reference_global_ids_ties_and_bad_noise():
    result = reference.recovered_tokens([1, 0, 1], [4, 5], [[0.5, 0.5, 0.1]] * 2,
                                        None, [[7, 4, 5]] * 2,
                                        [[1, 1, float("nan")], [1] * 3, [0, 1, float("inf")]])
    assert result == [7, 4]
    assert reference.recovered_tokens([1], [0], [[0.5, 0.5]], [[0.25, 0.0]], None, [[1, 1]]) == [1]


def test_penalty_reference_order_and_sign():
    assert reference.penalties([[2, -2, 0, 4]], [[True, False, False, False]], [[0, 2, 1, 0]],
                               [2], [0.25], [0.5]) == [[1, -5, -0.75, 4]]


def test_rotary_reference_layout_and_untouched_suffix():
    assert reference.rotary([1, 2, 3, 4, 99], [0.5, 0.5], [0.25, 0.25]) == [-0.25, 0, 1.75, 2.5, 99]
    assert reference.rotary([1, 2, 3, 4, 99], [0.5, 0.5], [0.25, 0.25], False) == [0, 1.25, 0.5, 2.75, 99]


def test_mrope_reference_channel_selection():
    assert reference.mrope_channels(8, (4, 2, 2), False) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert reference.mrope_channels(8, (4, 2, 2), True) == [0, 1, 2, 0, 1, 2, 0, 0]
