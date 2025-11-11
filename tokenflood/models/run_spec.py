from typing import List, Self, Tuple

from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from tokenflood.models.load_type import LoadType, NonEmptyLoadTypes


class RunSpec(BaseModel, frozen=True):
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


class HeuristicRunSpec(RunSpec, frozen=True):
    load_types: NonEmptyLoadTypes

    def sample_n(self, n: int) -> Tuple[List[int], List[int], List[int]]:
        loads = self.sample_loads(n)
        prompt_lengths, prefix_lengths, output_lengths = [], [], []
        for load in loads:
            prompt_lengths.append(load.prompt_length)
            prefix_lengths.append(load.prefix_length)
            output_lengths.append(load.output_length)
        return prompt_lengths, prefix_lengths, output_lengths

    def sample(self) -> Tuple[List[int], List[int], List[int]]:
        return self.sample_n(self.total_num_requests)

    def sample_loads(self, n: int) -> List[LoadType]:
        sampled: List[LoadType] = []
        while len(sampled) < n:
            for load_type in self.load_types:
                sampled.extend([load_type] * load_type.weight)
        return sampled[:n]
