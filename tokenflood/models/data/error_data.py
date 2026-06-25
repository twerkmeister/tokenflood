from pydantic import BaseModel

from tokenflood.models.validation_types import NonEmptyString, GroupID


class ErrorData(BaseModel, frozen=True):
    datetime: NonEmptyString
    request_per_second_phase: float
    type: str
    message: str
    group_id: GroupID


class ErrorContext(BaseModel, frozen=True):
    requests_per_second_phase: float
    group_id: GroupID
