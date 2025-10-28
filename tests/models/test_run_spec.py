import numpy as np
import pytest

from tests.utils import does_not_raise
from tokenflood.models.load_type import LoadType
from tokenflood.models.run_spec import HeuristicRunSpec, RunSpec


@pytest.mark.parametrize(
    "requests_per_second, test_length_in_seconds, expectation",
    [
        (3, 10, does_not_raise()),
        (0, 10, pytest.raises(ValueError)),
        (-1, 10, pytest.raises(ValueError)),
        (-3, 0, pytest.raises(ValueError)),
        (-3, -1, pytest.raises(ValueError)),
        (0.1, 1, pytest.raises(ValueError)),
    ],
)
def test_run_spec_validation(requests_per_second, test_length_in_seconds, expectation):
    with expectation:
        RunSpec(
            requests_per_second=requests_per_second,
            test_length_in_seconds=test_length_in_seconds,
        )


@pytest.mark.parametrize(
    "load_types, expectation",
    [
        (
            (
                LoadType(
                    prompt_length=1000, prefix_length=20, output_length=12, weight=1
                ),
            ),
            does_not_raise(),
        ),
        (
            (
                LoadType(
                    prompt_length=1000, prefix_length=20, output_length=12, weight=1
                ),
                LoadType(
                    prompt_length=1000, prefix_length=200, output_length=120, weight=1
                ),
            ),
            does_not_raise(),
        ),
        ([], pytest.raises(ValueError)),
    ],
)
def test_heuristic_run_spec_validation(load_types, expectation):
    with expectation:
        HeuristicRunSpec(
            requests_per_second=3,
            test_length_in_seconds=10,
            load_types=load_types,
        )


def test_heuristic_run_spec_sampling():
    spec = HeuristicRunSpec(
        requests_per_second=100,
        test_length_in_seconds=1000,
        load_types=(
            LoadType(prompt_length=100, prefix_length=20, output_length=12, weight=2),
            LoadType(prompt_length=112, prefix_length=20, output_length=6, weight=1),
        ),
    )

    prompt_lengths, prefix_lengths, output_lengths = spec.sample()
    assert np.allclose(np.average([100, 100, 112]), np.average(prompt_lengths), atol=1)
    assert np.allclose(20, np.average(prefix_lengths), atol=1)
    assert np.allclose(np.average([12, 12, 6]), np.average(output_lengths), atol=1)
