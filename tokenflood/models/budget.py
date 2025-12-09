from pydantic import BaseModel

from tokenflood.constants import MAX_INPUT_TOKENS_DEFAULT, MAX_OUTPUT_TOKENS_DEFAULT
from tokenflood.models.validation_types import PositiveInteger


class Budget(BaseModel, frozen=True):
    input_tokens: PositiveInteger = MAX_INPUT_TOKENS_DEFAULT
    output_tokens: PositiveInteger = MAX_OUTPUT_TOKENS_DEFAULT
