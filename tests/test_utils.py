import pytest
from datetime import datetime
from tokenflood.util import calculate_mean_absolute_error, get_run_name


def test_calculate_mean_absolute_error_sequence_length_mismatch():
    a, b = [1, 2], [1]
    with pytest.raises(ValueError):
        calculate_mean_absolute_error(a, b)


def test_get_run_name(base_endpoint_spec):
    run_name = get_run_name(base_endpoint_spec)
    current_date = datetime.now().strftime("%Y-%m-%d")
    assert run_name.startswith(current_date)
    assert run_name.endswith(base_endpoint_spec.provider_model_str_as_folder_name)
