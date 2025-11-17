import pytest

from tokenflood.heuristic import builtin_heuristic_tasks, builtin_heuristic_token_sets
from tokenflood.models.load_type import LoadType
from tokenflood.models.observation_spec import ObservationSpec
from tokenflood.observer import create_even_schedule, create_schedule


@pytest.fixture
def default_observation_spec():
    return ObservationSpec(name="test", duration_hours=24, polling_interval_minutes=20,
                           load_type=LoadType(prompt_length=1024, prefix_length=512, output_length=20),
                           num_requests=4, within_seconds=1.0, task=builtin_heuristic_tasks[0],
                           token_set=builtin_heuristic_token_sets[0], percentiles=[50, 90, 99])

@pytest.mark.parametrize("num_requests, within_seconds, expected_result", [
    (5, 1.0, [0.25, 0.25, 0.25, 0.25]),
    (4, 1.5, [0.5, 0.5, 0.5]),
    (1, 2.0, [])
])
def test_create_even_schedule(num_requests, within_seconds, expected_result):
    assert expected_result == create_even_schedule(num_requests, within_seconds)


@pytest.mark.parametrize("spec_updates, expected_result", [
    ({}, [0.33, 0.33, 0.33, 60 * 20 -1] * 3 * 24),
    ({"duration_hours": 1}, [0.33, 0.33, 0.33, 60 * 20 -1] * 3)
])
def test_create_schedule(spec_updates, expected_result, default_observation_spec):
    observation_spec = default_observation_spec.model_copy(update=spec_updates)
    assert create_schedule(observation_spec) == expected_result
