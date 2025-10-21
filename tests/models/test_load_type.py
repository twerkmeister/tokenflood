from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.load_type import LoadType


@pytest.fixture()
def default_load_type_kwargs() -> Dict:
    return LoadType(
        prompt_length=1000, prefix_length=100, output_length=12, weight=1
    ).model_dump()


@pytest.mark.parametrize(
    "kwargs_override, expectation",
    [
        ({}, does_not_raise()),
        ({"weight": 0}, does_not_raise()),
        ({"prefix_length": 0}, does_not_raise()),
        ({"prompt_length": 1000.5}, pytest.raises(ValueError)),
        ({"prompt_length": -1000}, pytest.raises(ValueError)),
        ({"prompt_length": 0}, pytest.raises(ValueError)),
        ({"prefix_length": 100.3}, pytest.raises(ValueError)),
        ({"prefix_length": -100}, pytest.raises(ValueError)),
        ({"output_length": -12}, pytest.raises(ValueError)),
        ({"output_length": 12.1}, pytest.raises(ValueError)),
        ({"output_length": 0}, pytest.raises(ValueError)),
        ({"weight": -1}, pytest.raises(ValueError)),
    ],
)
def test_load_type_validation(kwargs_override, expectation, default_load_type_kwargs):
    with expectation:
        LoadType(**{**default_load_type_kwargs, **kwargs_override})
