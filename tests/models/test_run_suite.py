from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.load_type import LoadType
from tokenflood.models.run_suite import HeuristicRunSuite


@pytest.fixture()
def default_run_suite_kwargs() -> Dict:
    return HeuristicRunSuite(
        name="ABC",
        requests_per_second_rates=(1, 2, 3, 4),
        test_length_in_seconds=30,
        load_types=(LoadType(prompt_length=1024, prefix_length=400, output_length=12),),
        percentiles=(50, 90, 98),
    ).model_dump()


@pytest.mark.parametrize(
    "kwargs_override, expectation",
    [
        ({}, does_not_raise()),
        ({"name": ""}, pytest.raises(ValueError)),
        ({"requests_per_second_rates": ()}, pytest.raises(ValueError)),
        ({"requests_per_second_rates": (-1, 2, 3, 4)}, pytest.raises(ValueError)),
        ({"requests_per_second_rates": (0, 2, 3, 4)}, pytest.raises(ValueError)),
        ({"requests_per_second_rates": (1, 1, 3, 4)}, pytest.raises(ValueError)),
        ({"test_length_in_seconds": -5}, pytest.raises(ValueError)),
        ({"test_length_in_seconds": 0}, pytest.raises(ValueError)),
        ({"load_types": ()}, pytest.raises(ValueError)),
        ({"percentiles": ()}, pytest.raises(ValueError)),
        ({"percentiles": (-50, 90)}, pytest.raises(ValueError)),
    ],
)
def test_run_suite_validation(kwargs_override, expectation, default_run_suite_kwargs):
    with expectation:
        HeuristicRunSuite(**{**default_run_suite_kwargs, **kwargs_override})


def test_create_run_specs(default_run_suite_kwargs):
    run_suite = HeuristicRunSuite(**default_run_suite_kwargs)

    run_specs = run_suite.create_run_specs()
    assert len(run_specs) == len(run_suite.requests_per_second_rates)

    assert run_specs[0].name == f"{run_suite.name}_001.00"
    assert all(
        [
            rs.test_length_in_seconds == run_suite.test_length_in_seconds
            for rs in run_specs
        ]
    )
    assert all([rs.load_types == run_suite.load_types for rs in run_specs])
