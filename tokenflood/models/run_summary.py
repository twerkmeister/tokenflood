from typing import Dict, List

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from tokenflood.models.validation_types import NonEmptyString


class LoadResult(BaseModel, frozen=True):
    requests_per_second: NonNegativeFloat
    mean_request_latency: NonNegativeFloat
    mean_network_latency: NonNegativeFloat
    percentile_latency: Dict[str, float]


class RunSummary(BaseModel, frozen=True):
    run_suite: NonEmptyString
    endpoint: NonEmptyString
    total_num_requests: NonNegativeInt
    mean_expected_input_tokens: NonNegativeInt
    mean_measured_input_tokens: NonNegativeInt
    relative_input_token_error: float
    mean_expected_output_tokens: NonNegativeInt
    mean_measured_output_tokens: NonNegativeInt
    relative_output_token_error: float
    mean_expected_prefix_tokens: NonNegativeInt
    mean_measured_prefix_tokens: NonNegativeInt
    relative_prefix_token_error: float
    load_results: List[LoadResult]

    @classmethod
    def create_empty(cls, run_suite: str, endpoint: str):
        return RunSummary(
            run_suite=run_suite,
            endpoint=endpoint,
            total_num_requests=0,
            mean_expected_input_tokens=0,
            mean_measured_input_tokens=0,
            mean_expected_output_tokens=0,
            mean_measured_output_tokens=0,
            mean_expected_prefix_tokens=0,
            mean_measured_prefix_tokens=0,
            relative_input_token_error=0.0,
            relative_output_token_error=0.0,
            relative_prefix_token_error=0.0,
            load_results=[],
        )
