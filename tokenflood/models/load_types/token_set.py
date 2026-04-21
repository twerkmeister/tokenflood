import random

from pydantic import BaseModel

from tokenflood.models.validation_types import AtLeastTwoUniqueStrings


class TokenSet(BaseModel, frozen=True):
    tokens: AtLeastTwoUniqueStrings

    def sample(self, n: int):
        return random.choices(self.tokens, k=n)


DEFAULT_TOKEN_SET = TokenSet(tokens=tuple([" " + chr(c) for c in range(65, 91)]))  # " A" - " Z"