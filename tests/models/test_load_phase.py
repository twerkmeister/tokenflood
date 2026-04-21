import pytest

from tests.utils import does_not_raise
from tokenflood.models.run_specs.load_spec import LoadPhase


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
        LoadPhase(
            requests_per_second=requests_per_second,
            duration_seconds=test_length_in_seconds,
        )
