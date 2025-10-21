from pydantic import BaseModel, NonNegativeFloat

from tokenflood.models.validation_types import NonNegativeInteger, PositiveInteger


class LoadType(BaseModel, frozen=True):
    prompt_length: PositiveInteger
    prefix_length: NonNegativeInteger
    output_length: PositiveInteger
    weight: NonNegativeFloat = 1.0
