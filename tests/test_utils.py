import pytest
from datetime import datetime
from tokenflood.util import (
    calculate_mean_absolute_error,
    calculate_relative_error,
    get_run_name,
)


def test_calculate_mean_absolute_error_sequence_length_mismatch():
    a, b = [1, 2], [1]
    with pytest.raises(ValueError):
        calculate_mean_absolute_error(a, b)


@pytest.mark.parametrize(
    "s1, target, expected_result",
    [
        ([20], [80], 0.75),
        ([20, 40], [100, 100], 0.7),
        ([100], [100], 0.0),
        ([107], [100], 0.07),
        ([93], [100], 0.07),
    ],
)
def test_calculate_relative_error(s1, target, expected_result):
    assert calculate_relative_error(s1, target) == expected_result


def test_get_run_name(base_endpoint_spec):
    run_name = get_run_name(base_endpoint_spec)
    current_date = datetime.now().strftime("%Y-%m-%d")
    assert run_name.startswith(current_date)
    assert run_name.endswith(base_endpoint_spec.provider_model_str_as_folder_name)
