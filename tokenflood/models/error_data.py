from pydantic import BaseModel

from tokenflood.models.validation_types import NonEmptyString


class ErrorData(BaseModel, frozen=True):
    datetime: NonEmptyString
    request_per_second_phase: float
    type: str
    message: str


class ErrorContext(BaseModel, frozen=True):
    requests_per_second_phase: float
