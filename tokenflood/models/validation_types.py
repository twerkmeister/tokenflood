from typing import Annotated, Tuple

from pydantic import AfterValidator, Field, StringConstraints

from tokenflood.models.validators import (
    all_non_empty_strings,
    all_positive,
    all_strictly_positive,
    at_least_size,
    non_empty_list,
    unique_elements,
)

PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
NonEmptyStrings = Annotated[
    Tuple[str, ...],
    AfterValidator(non_empty_list),
    AfterValidator(all_non_empty_strings),
]
NonEmptyUniqueStrings = Annotated[
    Tuple[str, ...],
    AfterValidator(non_empty_list),
    AfterValidator(all_non_empty_strings),
    AfterValidator(unique_elements),
]
AtLeastTwoUniqueStrings = Annotated[
    Tuple[str, ...],
    AfterValidator(at_least_size(2)),
    AfterValidator(unique_elements),
    AfterValidator(all_non_empty_strings),
]
PositiveIntegers = Annotated[
    Tuple[int, ...],
    AfterValidator(non_empty_list),
    AfterValidator(all_strictly_positive),
]
PositiveUniqueIntegers = Annotated[
    Tuple[int, ...],
    AfterValidator(non_empty_list),
    AfterValidator(unique_elements),
    AfterValidator(all_strictly_positive),
]
PositiveFloats = Annotated[
    Tuple[float, ...],
    AfterValidator(non_empty_list),
    AfterValidator(all_strictly_positive),
]
PositiveUniqueFloats = Annotated[
    Tuple[float, ...],
    AfterValidator(non_empty_list),
    AfterValidator(unique_elements),
    AfterValidator(all_strictly_positive),
]
NonNegativeIntegers = Annotated[
    Tuple[int, ...], AfterValidator(non_empty_list), AfterValidator(all_positive)
]

NonNegativeIntegersOrEmpty = Annotated[Tuple[int, ...], AfterValidator(all_positive)]
