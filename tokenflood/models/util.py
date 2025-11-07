from typing import List, Type, Union

from pydantic import BaseModel

numeric = Union[
    int,
    float,
]


def get_fields(model_type: Type[BaseModel]) -> List[str]:
    return list(model_type.model_fields.keys())
