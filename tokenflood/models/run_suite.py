from typing import List

from pydantic import BaseModel, NonNegativeFloat

from tokenflood.constants import (
    DEFAULT_ERROR_RATE_LIMIT,
    MAX_INPUT_TOKENS_DEFAULT,
    MAX_OUTPUT_TOKENS_DEFAULT,
)
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import NonEmptyLoadTypes
from tokenflood.models.run_spec import HeuristicRunSpec
from tokenflood.models.token_set import TokenSet
from tokenflood.models.validation_types import (
    NonEmptyString,
    PositiveInteger,
    PositiveUniqueFloats,
    PositiveUniqueIntegers,
)


class HeuristicRunSuite(BaseModel, frozen=True):
    name: NonEmptyString
    requests_per_second_rates: PositiveUniqueFloats
    test_length_in_seconds: PositiveInteger
    load_types: NonEmptyLoadTypes
    percentiles: PositiveUniqueIntegers
    task: HeuristicTask
    token_set: TokenSet
    error_limit: NonNegativeFloat = DEFAULT_ERROR_RATE_LIMIT
    input_token_budget: PositiveInteger = MAX_INPUT_TOKENS_DEFAULT
    output_token_budget: PositiveInteger = MAX_OUTPUT_TOKENS_DEFAULT

    def create_run_specs(self) -> List[HeuristicRunSpec]:
        return [
            HeuristicRunSpec(
                requests_per_second=rate,
                test_length_in_seconds=self.test_length_in_seconds,
                load_types=self.load_types,
            )
            for rate in self.requests_per_second_rates
        ]
