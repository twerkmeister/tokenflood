from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.load_types.load_type import HeuristicLoad
from tokenflood.models.run_specs.load_spec import LoadSpec


@pytest.fixture()
def default_run_suite_kwargs() -> Dict:
    return LoadSpec(
        name="ABC",
        requests_per_second_phases=(1, 2, 3, 4),
        seconds_per_phase=30,
        load_type=HeuristicLoad(prompt_length=1024, prefix_length=400, output_length=12)
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
def test_load_spec_validation(kwargs_override, expectation, default_run_suite_kwargs):
    with expectation:
        LoadSpec(**{**default_run_suite_kwargs, **kwargs_override})


def test_create_load_phases(default_run_suite_kwargs):
    load_spec = LoadSpec(**default_run_suite_kwargs)

    load_phases = load_spec.create_load_phases()
    assert len(load_phases) == len(load_spec.requests_per_second_phases)

    assert all(
        [
            rs.duration_seconds == load_spec.seconds_per_phase
            for rs in load_phases
        ]
    )

