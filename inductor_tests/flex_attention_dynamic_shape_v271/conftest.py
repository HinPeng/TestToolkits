"""Shared pytest isolation for Flex Attention dynamic-shape tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo_state():
    """Keep compile-cache and compile-count assertions test-local."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
