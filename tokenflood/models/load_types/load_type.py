import random
from typing import Literal, Annotated

from pydantic import BaseModel, Field

from tokenflood.constants import DEFAULT_HEURISTIC_TASK, DEFAULT_PROMPT_FILLER_TOKENS
from tokenflood.models.messages import MessageList
from tokenflood.models.validation_types import (
    NonNegativeInteger,
    PositiveInteger,
    AtLeastTwoUniqueStrings,
    NonEmptyString,
)
from tokenflood.util import roughly_estimated_token_cost


class LoadType(BaseModel, frozen=True):
    type: str

    def create_prompts(self, n: int) -> list[str]:
        raise NotImplementedError

    def create_message_lists(self, n: int) -> list[MessageList]:
        raise NotImplementedError

    def get_expected_prompt_length(self) -> int:
        raise NotImplementedError

    def get_expected_prefix_length(self) -> int:
        raise NotImplementedError

    def get_expected_output_length(self) -> int:
        raise NotImplementedError


class HeuristicLoad(LoadType, frozen=True):
    type: Literal["heuristic"] = "heuristic"
    prompt_length: PositiveInteger
    prefix_length: NonNegativeInteger
    output_length: PositiveInteger
    task: NonEmptyString = DEFAULT_HEURISTIC_TASK
    prompt_filler_tokens: AtLeastTwoUniqueStrings = DEFAULT_PROMPT_FILLER_TOKENS

    def create_prompts(self, n: int) -> list[str]:
        return [self.create_prompt() for _ in range(n)]

    def create_prompt(self) -> str:
        """Create a prompt that suffices the length constraints.

        The prompt is structured like this:

        1. Common Prefix (using the prompt filler tokens)
        2. Random Tokens (using the prompt filler tokens)
        3. A single newline to separate the task from the random part
        4. The Task
        """
        task_tokens = roughly_estimated_token_cost(self.task)
        random_prompt_tokens = self.prompt_length - self.prefix_length - task_tokens
        prompt = ""
        prompt += self.create_prompt_prefix(self.prefix_length)
        if random_prompt_tokens > 0:
            prompt += self.create_prompt_random_part(random_prompt_tokens)
        if len(prompt) > 0:
            prompt += "\n\n"
        prompt += self.task
        return prompt

    def create_prompt_prefix(self, length: int) -> str:
        """Create a predictable prefix for prompts."""
        return self.prompt_filler_tokens[0] * max(0, length)

    def create_prompt_random_part(self, length: int) -> str:
        """Create the random prompt part between prefix and task."""
        return "".join(random.choices(self.prompt_filler_tokens, k=length))

    def create_message_lists(self, n: int) -> list[MessageList]:
        return [
            [{"role": "user", "content": prompt}] for prompt in self.create_prompts(n)
        ]

    def get_expected_prompt_length(self) -> int:
        return self.prompt_length

    def get_expected_output_length(self) -> int:
        return self.output_length

    def get_expected_prefix_length(self) -> int:
        return self.prefix_length


SpecificLoadType = Annotated[HeuristicLoad, Field(discriminator="type")]
