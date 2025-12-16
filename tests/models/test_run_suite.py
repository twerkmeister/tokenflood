from typing import Dict

import numpy as np
import pytest

from tests.utils import does_not_raise
from tokenflood.heuristic import builtin_heuristic_tasks, builtin_heuristic_token_sets
from tokenflood.models.load_type import LoadType
from tokenflood.models.run_suite import HeuristicRunSuite


@pytest.fixture()
def default_run_suite_kwargs() -> Dict:
    return HeuristicRunSuite(
        name="ABC",
        requests_per_second_rates=(1, 2, 3, 4),
        test_length_in_seconds=30,
        load_types=(LoadType(prompt_length=1024, prefix_length=400, output_length=12),),
        task=builtin_heuristic_tasks[0],
        token_set=builtin_heuristic_token_sets[0],
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
    ],
)
def test_run_suite_validation(kwargs_override, expectation, default_run_suite_kwargs):
    with expectation:
        HeuristicRunSuite(**{**default_run_suite_kwargs, **kwargs_override})


def test_create_run_specs(default_run_suite_kwargs):
    run_suite = HeuristicRunSuite(**default_run_suite_kwargs)

    run_specs = run_suite.create_run_specs()
    assert len(run_specs) == len(run_suite.requests_per_second_rates)

    assert all(
        [
            rs.test_length_in_seconds == run_suite.test_length_in_seconds
            for rs in run_specs
        ]
    )
    assert all([rs.load_types == run_suite.load_types for rs in run_specs])


def test_estimate_token_usage_tiny(tiny_run_suite):
    estimated_input_tokens, estimated_output_tokens = (
        tiny_run_suite.get_input_output_token_cost()
    )
    num_requests = 8
    assert estimated_input_tokens == 256 * num_requests
    assert estimated_output_tokens == 2 * num_requests


def test_estimate_token_usage_base(base_run_suite):
    estimated_input_tokens, estimated_output_tokens = (
        base_run_suite.get_input_output_token_cost()
    )
    num_requests = 300
    input_tokens_diff = abs(
        estimated_input_tokens - np.average([1024, 1024, 1200]) * num_requests
    )
    # less than 1% diff
    assert input_tokens_diff / estimated_input_tokens < 0.01

    output_tokens_diff = abs(
        estimated_output_tokens - np.average([16, 16, 40]) * num_requests
    )
    assert output_tokens_diff / estimated_output_tokens < 0.05
