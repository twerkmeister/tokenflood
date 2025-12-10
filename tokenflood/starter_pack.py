from tokenflood.constants import DEFAULT_ERROR_RATE_LIMIT
from tokenflood.heuristic import builtin_heuristic_tasks, builtin_heuristic_token_sets
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.load_type import LoadType
from tokenflood.models.observation_spec import ObservationSpec
from tokenflood.models.run_suite import HeuristicRunSuite

starter_run_suite = HeuristicRunSuite(
    name="starter",
    requests_per_second_rates=(1, 2),
    test_length_in_seconds=30,
    load_types=(
        LoadType(prompt_length=512, prefix_length=128, output_length=32, weight=1),
        LoadType(prompt_length=640, prefix_length=568, output_length=12, weight=1),
    ),
    percentiles=(50, 90, 99),
    task=builtin_heuristic_tasks[0],
    token_set=builtin_heuristic_token_sets[0],
    error_limit=DEFAULT_ERROR_RATE_LIMIT,
)

starter_model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
starter_endpoint_spec_vllm = EndpointSpec(
    provider="hosted_vllm", model=starter_model_id, base_url="http://127.0.0.1:8000/v1"
)

starter_observation_spec = ObservationSpec(
    name="starter",
    duration_hours=1.0,
    polling_interval_minutes=15,
    load_type=LoadType(prompt_length=512, prefix_length=128, output_length=32),
    num_requests=5,
    within_seconds=2.0,
    task=builtin_heuristic_tasks[0],
    token_set=builtin_heuristic_token_sets[0],
    percentiles=(50, 90, 99),
)
