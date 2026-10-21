from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.load_types.load_type import HeuristicLoad


@pytest.fixture()
def default_heuristic_load_kwargs() -> Dict:
    return HeuristicLoad(
        prompt_length=1000,
        prefix_length=100,
        output_length=12,
    ).model_dump()


@pytest.mark.parametrize(
    "kwargs_override, expectation",
    [
        ({}, does_not_raise()),
        ({"prefix_length": 0}, does_not_raise()),
        ({"prompt_length": 1000.5}, pytest.raises(ValueError)),
        ({"prompt_length": -1000}, pytest.raises(ValueError)),
        ({"prompt_length": 0}, pytest.raises(ValueError)),
        ({"prefix_length": 100.3}, pytest.raises(ValueError)),
        ({"prefix_length": -100}, pytest.raises(ValueError)),
        ({"output_length": -12}, pytest.raises(ValueError)),
        ({"output_length": 12.1}, pytest.raises(ValueError)),
        ({"output_length": 0}, pytest.raises(ValueError)),
        ({"task": ""}, pytest.raises(ValueError)),
        ({"prompt_filler_tokens": ("A",)}, pytest.raises(ValueError)),
        ({"prompt_filler_tokens": ("A", "A")}, pytest.raises(ValueError)),
    ],
)
def test_heuristic_load_validation(
    kwargs_override, expectation, default_heuristic_load_kwargs
):
    with expectation:
        HeuristicLoad(**{**default_heuristic_load_kwargs, **kwargs_override})


def test_dynamic_prompt_based_load_(prompt_based_dynamic_load_test_spec):
    n = 28
    prompts = prompt_based_dynamic_load_test_spec.load_type.create_message_lists(n)
    assert len(prompts) == n
    assert all([len(prompt) == 1 for prompt in prompts])
    contents = [prompt[0]["content"] for prompt in prompts]
    assert "dynamic_part: A B" in contents[-1]
    assert "dynamic_part: A" in contents[0]


def test_static_prompt_based_load(prompt_based_static_load_test_spec):
    n = 28
    prompts = prompt_based_static_load_test_spec.load_type.create_message_lists(n)
    assert len(prompts) == n
    assert all([len(prompt) == 1 for prompt in prompts])
    contents = [prompt[0]["content"] for prompt in prompts]
    # all identical
    assert all([content == contents[0] for content in contents])
