from tokenflood.constants import (
    DEFAULT_ERROR_RATE_LIMIT,
    DEFAULT_HEURISTIC_TASK,
    DEFAULT_PROMPT_FILLER_TOKENS,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.load_types.load_type import HeuristicLoad
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.models.run_specs.load_spec import LoadSpec

starter_run_suite = LoadSpec(
    name="starter",
    requests_per_second_phases=(1, 2),
    seconds_per_phase=30,
    load_type=HeuristicLoad(
        prompt_length=512,
        prefix_length=128,
        output_length=32,
        task=DEFAULT_HEURISTIC_TASK,
        prompt_filler_tokens=DEFAULT_PROMPT_FILLER_TOKENS,
    ),
    error_limit=DEFAULT_ERROR_RATE_LIMIT,
)

starter_model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
starter_endpoint_spec_vllm = EndpointSpec(
    provider="hosted_vllm",
    name="local_vllm",
    model=starter_model_id,
    base_url="http://127.0.0.1:8000/v1",
)

starter_observation_spec = ObservationSpec(
    name="starter",
    duration_hours=1.0,
    polling_interval_minutes=15,
    load_type=HeuristicLoad(
        prompt_length=512,
        prefix_length=128,
        output_length=32,
        task=DEFAULT_HEURISTIC_TASK,
        prompt_filler_tokens=DEFAULT_PROMPT_FILLER_TOKENS,
    ),
    num_requests=5,
    within_seconds=2.0,
)
