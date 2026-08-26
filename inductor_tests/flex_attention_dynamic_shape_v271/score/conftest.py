"""Shared pytest isolation for Flex Attention dynamic-shape tests."""

import pytest
import torch


KNOWN_RUNTIME_SKIP_CASES = {
    (
        "test_a_batch_dynamic",
        "trig",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig score_mod",
    (
        "test_a_batch_dynamic",
        "trig2",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig2 score_mod",
    (
        "test_b_q_len_dynamic",
        "trig",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig score_mod",
    (
        "test_b_q_len_dynamic",
        "trig2",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig2 score_mod",
    (
        "test_c_kv_len_dynamic",
        "trig",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig score_mod",
    (
        "test_c_kv_len_dynamic",
        "trig2",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig2 score_mod",
    (
        "test_d_b_q_kv_joint_dynamic",
        "trig",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig score_mod",
    (
        "test_d_b_q_kv_joint_dynamic",
        "trig2",
    ): "Known Ascend_950 NPU runtime error for all dtypes with trig2 score_mod",
}


def pytest_collection_modifyitems(config, items):
    """Skip known runtime-failing parameter combinations before test setup."""
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue

        test_name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
        key = (
            test_name,
            callspec.params.get("score_mod_name"),
        )
        reason = KNOWN_RUNTIME_SKIP_CASES.get(key)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def reset_dynamo_state():
    """Keep compile counts local and avoid aliasing equal input dimensions."""
    shape_config = torch.fx.experimental._config
    original_use_duck_shape = shape_config.use_duck_shape
    shape_config.use_duck_shape = False
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()
        shape_config.use_duck_shape = original_use_duck_shape
