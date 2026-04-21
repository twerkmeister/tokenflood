from typing import Literal, Annotated

from pydantic import BaseModel, Field

from tokenflood.models.load_types.heuristic_task import (
    HeuristicTask,
    DEFAULT_HEURISTIC_TASK,
)
from tokenflood.models.messages import MessageList
from tokenflood.models.load_types.token_set import TokenSet, DEFAULT_TOKEN_SET
from tokenflood.models.validation_types import NonNegativeInteger, PositiveInteger


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
    task: HeuristicTask = DEFAULT_HEURISTIC_TASK
    token_set: TokenSet = DEFAULT_TOKEN_SET

    def create_prompts(self, n: int) -> list[str]:
        return [self.create_prompt() for _ in range(n)]

    def create_prompt(self) -> str:
        """Create a prompt that suffices the length constraints.

        The prompt is structured like this:

        1. Common Prefix (using the TokenSet)
        2. Random Tokens (using the TokenSet)
        3. A single newline to separate the task from the random part
        4. The Task
        """
        task_tokens = self.task.roughly_estimated_token_cost
        random_prompt_tokens = self.prompt_length - self.prefix_length - task_tokens
        prompt = ""
        prompt += self.create_prompt_prefix(self.prefix_length)
        if random_prompt_tokens > 0:
            prompt += self.create_prompt_random_part(random_prompt_tokens)
        if len(prompt) > 0:
            prompt += "\n\n"
        prompt += self.task.task
        return prompt

    def create_prompt_prefix(self, length: int) -> str:
        """Create a predictable prefix for prompts."""
        return self.token_set.tokens[0] * max(0, length)

    def create_prompt_random_part(self, length: int) -> str:
        """Create the random prompt part between prefix and task."""
        return "".join(self.token_set.sample(length))

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
