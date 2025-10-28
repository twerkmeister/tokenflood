import pytest
from datetime import datetime

from litellm.types.utils import ModelResponse

from tokenflood.runner import make_empty_response
from tokenflood.util import (
    calculate_mean_error,
    calculate_relative_error,
    find_idx,
    get_run_name,
)


def test_calculate_mean_error_sequence_length_mismatch():
    a, b = [1, 2], [1]
    with pytest.raises(ValueError):
        calculate_mean_error(a, b)


@pytest.mark.parametrize(
    "observations, targets, expected_result",
    [
        ([20], [80], -60),
        ([20, 40], [100, 100], -70),
        ([100], [100], 0.0),
        ([107], [100], 7),
        ([93], [100], -7),
    ],
)
def test_calculate_mean_error(observations, targets, expected_result):
    assert calculate_mean_error(observations, targets) == expected_result


@pytest.mark.parametrize(
    "observations, targets, expected_result",
    [
        ([20], [80], -0.75),
        ([20, 40], [100, 100], -0.7),
        ([100], [100], 0.0),
        ([107], [100], 0.07),
        ([93], [100], -0.07),
    ],
)
def test_calculate_relative_error(observations, targets, expected_result):
    assert calculate_relative_error(observations, targets) == expected_result


def test_get_run_name(base_endpoint_spec):
    run_name = get_run_name(base_endpoint_spec)
    current_date = datetime.now().strftime("%Y-%m-%d")
    assert run_name.startswith(current_date)
    assert run_name.endswith(base_endpoint_spec.provider_model_str_as_folder_name)


@pytest.mark.parametrize(
    "s, predicate, expected_result",
    [
        ([0, 1, 2], lambda x: x > 1, 2),
        ([0, 1, 2], lambda x: x > 2, None),
        ([0, 1, 2], lambda x: True, 0),
        ([], lambda x: True, None),
        (
            [make_empty_response(), ValueError("ABC")],
            lambda x: not isinstance(x, ModelResponse),
            1,
        ),
    ],
)
def test_find_idx(s, predicate, expected_result):
    assert find_idx(s, predicate) == expected_result
