from pydantic import BaseModel

from tokenflood.models.load_type import NonEmptyLoadTypes
from tokenflood.models.validation_types import (
    NonEmptyString,
    PositiveFloats,
    PositiveInteger,
    PositiveIntegers,
)


class HeuristicRunSuite(BaseModel, frozen=True):
    name: NonEmptyString
    requests_per_second_rates: PositiveFloats
    test_length_in_seconds: PositiveInteger
    load_types: NonEmptyLoadTypes
    percentiles: PositiveIntegers
