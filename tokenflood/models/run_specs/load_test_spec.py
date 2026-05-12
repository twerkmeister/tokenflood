from typing import List, Literal, Self

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from tokenflood.constants import DEFAULT_ERROR_RATE_LIMIT, LOAD_TEST_SPEC_FILE
from tokenflood.models.load_types.load_type import SpecificLoadType
from tokenflood.models.run_specs.run_spec import RunSpec
from tokenflood.models.validation_types import (
    NonEmptyString,
    PositiveInteger,
    PositiveUniqueFloats,
)


class LoadTestPhase(BaseModel, frozen=True):
    requests_per_second: PositiveFloat
    duration_seconds: PositiveInt

    @property
    def total_num_requests(self) -> int:
        return int(self.requests_per_second * self.duration_seconds)

    @model_validator(mode="after")
    def check_has_at_least_one_request(self) -> Self:
        if self.total_num_requests < 1:
            raise ValueError(
                "Total number of requests "
                "(=int(requests_per_second * test_length_in_seconds)) for "
                f"this test would be {self.total_num_requests}. It needs to have at least one request"
            )
        return self


class LoadTestSpec(RunSpec, frozen=True):
    type: Literal["load_test"] = "load_test"
    name: NonEmptyString
    requests_per_second_phases: PositiveUniqueFloats
    seconds_per_phase: PositiveInteger
    load_type: SpecificLoadType
    burstiness: int = Field(ge=0, le=10, default=1)
    error_limit: NonNegativeFloat = DEFAULT_ERROR_RATE_LIMIT

    def create_load_test_phases(self) -> List[LoadTestPhase]:
        return [
            LoadTestPhase(
                requests_per_second=rate,
                duration_seconds=self.seconds_per_phase,
            )
            for rate in self.requests_per_second_phases
        ]

    @property
    def total_seconds(self) -> int:
        return len(self.requests_per_second_phases) * self.seconds_per_phase

    @property
    def run_spec_file(self) -> str:
        return LOAD_TEST_SPEC_FILE

    @property
    def total_num_requests(self) -> int:
        return sum(
            [phase.total_num_requests for phase in self.create_load_test_phases()]
        )
