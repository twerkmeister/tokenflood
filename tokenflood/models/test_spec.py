from typing import List, Self, Tuple
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


class HeuristicTestSpec(TestSpec, frozen=True):
    prompt_lengths: StrictlyPositiveIntegers
    output_lengths: StrictlyPositiveIntegers
    prefix_lengths: NonNegativeIntegers = [0]  # pydantic magic on mutable defaults

    def sample_n(self, n: int) -> Tuple[List[int], List[int], List[int]]:
        return (
            random.choices(self.prompt_lengths, k=n),
            random.choices(self.prefix_lengths, k=n),
            random.choices(self.output_lengths, k=n),
        )

    def sample(self) -> Tuple[List[int], List[int], List[int]]:
        return self.sample_n(self.total_num_requests)
