import pytest

from tokenflood.heuristic import builtin_heuristic_tasks, builtin_heuristic_token_sets
from tokenflood.models.load_type import LoadType
from tokenflood.models.observation_spec import ObservationSpec


@pytest.fixture
def default_observation_spec():
    return ObservationSpec(name="test", duration_hours=24, polling_interval_minutes=20,
                           load_type=LoadType(prompt_length=1024, prefix_length=512, output_length=20),
                           num_requests=4, within_seconds=1.0, task=builtin_heuristic_tasks[0],
                           token_set=builtin_heuristic_token_sets[0], percentiles=[50, 90, 99])

@pytest.mark.parametrize("spec_update, expected_result", [
    ({}, 24*3),
    ({"duration_hours": 12}, 12 * 3),
    ({"polling_interval_minutes": 1}, 24 * 60)
])
def test_num_polls(spec_update, expected_result, default_observation_spec):
    observation_spec = default_observation_spec.model_copy(update=spec_update)
    assert observation_spec.num_polls() == expected_result


@pytest.mark.parametrize("spec_update, expected_result", [
    ({}, 24*3*4),
    ({"duration_hours": 12}, 12 * 3 * 4),
    ({"polling_interval_minutes": 1}, 24 * 60 * 4)
])
def test_total_num_requests(spec_update, expected_result, default_observation_spec):
    observation_spec = default_observation_spec.model_copy(update=spec_update)
    assert observation_spec.total_num_requests() == expected_result
