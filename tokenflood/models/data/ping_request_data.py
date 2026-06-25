from typing import Self

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from tokenflood.models.validation_types import NonEmptyString, GroupID


class PingRequestContext(BaseModel, frozen=True):
    datetime: NonEmptyString
    endpoint_url: NonEmptyString
    requests_per_second_phase: NonNegativeFloat
    group_id: GroupID


class PingData(BaseModel, frozen=True):
    datetime: NonEmptyString
    endpoint_url: NonEmptyString
    requests_per_second_phase: NonNegativeFloat
    latency: NonNegativeInt
    group_id: GroupID

    @classmethod
    def from_context(cls, context_data: PingRequestContext, latency: int) -> Self:
        return cls(latency=latency, **context_data.model_dump())

    class F:
        latency = "latency"
