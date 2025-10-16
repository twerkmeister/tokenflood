import math

from pydantic import BaseModel
from tokenizers import Tokenizer

from tokenflood.models.validation_types import NonEmptyString


class HeuristicTask(BaseModel, frozen=True):
    task: NonEmptyString

    @property
    def roughly_estimated_token_cost(self) -> int:
        return math.ceil(len(self.task) / 3.5)

    def get_token_cost(self, tokenizer: Tokenizer) -> int:
        res = tokenizer.encode(self.task)
        return len(res.ids)
