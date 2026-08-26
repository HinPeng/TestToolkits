"""Shared pytest isolation for Flex Attention dynamic-shape tests."""

import pytest
import torch


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
