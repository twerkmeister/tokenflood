from typing import List, Self, Sequence, Tuple
import random
from pydantic import BaseModel, PositiveFloat, PositiveInt, model_validator

from tokenflood.models.validation_types import (
    NonEmptyString,
    NonNegativeIntegers,
    StrictlyPositiveIntegers,
)


class TestSpec(BaseModel, frozen=True):
    name: NonEmptyString
    requests_per_second: PositiveFloat
    test_length_in_seconds: PositiveInt

    @property
    def total_num_requests(self) -> int:
        return int(self.requests_per_second * self.test_length_in_seconds)

    @model_validator(mode="after")
    def check_test_has_at_least_one_request(self) -> Self:
        if self.total_num_requests == 0:
            raise ValueError(
                "Total number of requests "
                "(=int(requests_per_second * test_length_in_seconds)) for "
                "this test would be 0."
            )
        return self


class HeuristicTestSpec(TestSpec, frozen=True):
    prompt_lengths: StrictlyPositiveIntegers
    output_lengths: StrictlyPositiveIntegers
    prefix_lengths: NonNegativeIntegers = (0,)

    def sample_n_with(self, sequences: List[Sequence[int]], n: int) -> List[List[int]]:
        return [random.choices(s, k=n) for s in sequences]

    def sample_output_tokens(self) -> List[int]:
        return self.sample_n_with([self.output_lengths],
                                  self.total_num_requests)[0]

    def sample_input_tokens(self) -> Tuple[List[int], List[int]]:
        res = self.sample_n_with([self.prompt_lengths, self.prefix_lengths],
                                 self.total_num_requests)
        return res[0], res[1]
