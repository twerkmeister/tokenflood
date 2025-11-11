from typing import Annotated, Tuple

from pydantic import AfterValidator, BaseModel

from tokenflood.models.validation_types import NonNegativeInteger, PositiveInteger
from tokenflood.models.validators import non_empty_list


class LoadType(BaseModel, frozen=True):
    prompt_length: PositiveInteger
    prefix_length: NonNegativeInteger
    output_length: PositiveInteger
    weight: NonNegativeInteger = 1


NonEmptyLoadTypes = Annotated[Tuple[LoadType, ...], AfterValidator(non_empty_list)]
