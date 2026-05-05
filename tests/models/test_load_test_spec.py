from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.load_types.load_type import HeuristicLoad
from tokenflood.models.run_specs.load_test_spec import LoadTestSpec


@pytest.fixture()
def default_load_test_spec_kwargs() -> Dict:
    return LoadTestSpec(
        name="ABC",
        requests_per_second_phases=(1, 2, 3, 4),
        seconds_per_phase=30,
        load_type=HeuristicLoad(
            prompt_length=1024, prefix_length=400, output_length=12
        ),
    ).model_dump()


@pytest.mark.parametrize(
    "kwargs_override, expectation",
    [
        ({}, does_not_raise()),
        ({"name": ""}, pytest.raises(ValueError)),
        ({"requests_per_second_phases": ()}, pytest.raises(ValueError)),
        ({"requests_per_second_phases": (-1, 2, 3, 4)}, pytest.raises(ValueError)),
        ({"requests_per_second_phases": (0, 2, 3, 4)}, pytest.raises(ValueError)),
        ({"requests_per_second_phases": (1, 1, 3, 4)}, pytest.raises(ValueError)),
        ({"seconds_per_phase": -5}, pytest.raises(ValueError)),
        ({"seconds_per_phase": 0}, pytest.raises(ValueError)),
        ({"load_type": None}, pytest.raises(ValueError)),
    ],
)
def test_load_test_spec_validation(
    kwargs_override, expectation, default_load_test_spec_kwargs
):
    with expectation:
        LoadTestSpec(**{**default_load_test_spec_kwargs, **kwargs_override})


def test_create_load_test_phases(default_load_test_spec_kwargs):
    load_test_spec = LoadTestSpec(**default_load_test_spec_kwargs)

    load_phases = load_test_spec.create_load_test_phases()
    assert len(load_phases) == len(load_test_spec.requests_per_second_phases)

    assert all(
        [rs.duration_seconds == load_test_spec.seconds_per_phase for rs in load_phases]
    )
