from typing import List, Tuple

from pydantic import BaseModel, NonNegativeFloat

from tokenflood.constants import DEFAULT_ERROR_RATE_LIMIT
from tokenflood.models.budget import Budget
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import NonEmptyLoadTypes
from tokenflood.models.run_spec import HeuristicRunSpec
from tokenflood.models.token_cost_aware import TokenCostAware
from tokenflood.models.token_set import TokenSet
from tokenflood.models.validation_types import (
    NonEmptyString,
    PositiveInteger,
    PositiveUniqueFloats,
    PositiveUniqueIntegers,
)


class HeuristicRunSuite(BaseModel, TokenCostAware, frozen=True):
    name: NonEmptyString
    requests_per_second_rates: PositiveUniqueFloats
    test_length_in_seconds: PositiveInteger
    load_types: NonEmptyLoadTypes
    percentiles: PositiveUniqueIntegers
    task: HeuristicTask
    token_set: TokenSet
    error_limit: NonNegativeFloat = DEFAULT_ERROR_RATE_LIMIT
    budget: Budget = Budget()

    def create_run_specs(self) -> List[HeuristicRunSpec]:
        return [
            HeuristicRunSpec(
                requests_per_second=rate,
                test_length_in_seconds=self.test_length_in_seconds,
                load_types=self.load_types,
            )
            for rate in self.requests_per_second_rates
        ]

    def get_input_output_token_cost(self) -> Tuple[int, int]:
        """Estimate total token usage based on the run suite parameters.

        Specifically: requests per seconds, length of test, load types."""
        total_input_tokens = 0
        total_output_tokens = 0
        for run_spec in self.create_run_specs():
            input_tokens, _, output_tokens = run_spec.sample()
            total_input_tokens += sum(input_tokens)
            total_output_tokens += sum(output_tokens)
        return total_input_tokens, total_output_tokens
