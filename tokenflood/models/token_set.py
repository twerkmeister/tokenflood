import random

from pydantic import BaseModel

from tokenflood.models.validation_types import AtLeastTwoUniqueStrings


class TokenSet(BaseModel, frozen=True):
    tokens: AtLeastTwoUniqueStrings

    def sample(self, n: int):
        return random.choices(self.tokens, k=n)
