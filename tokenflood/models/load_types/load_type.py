import random
from typing import Literal, Annotated, Any, Generator

from pydantic import BaseModel, Field, PrivateAttr

from tokenflood.constants import DEFAULT_HEURISTIC_TASK, DEFAULT_PROMPT_FILLER_TOKENS
from tokenflood.io import read_prompts
from tokenflood.messages import (
    create_message_list_from_prompt,
    split_off_last_assistant_answer,
    inject_into_prompt,
)
from tokenflood.models.message_list import MessageList
from tokenflood.models.validation_types import (
    NonNegativeInteger,
    PositiveInteger,
    AtLeastTwoUniqueStrings,
    NonEmptyString,
)
from tokenflood.util import (
    roughly_estimated_token_cost,
    sample_exhaustively,
    sample_unique_concatenations_exhaustively,
)


class LoadType(BaseModel, frozen=True):
    type: str

    def create_message_lists(self, n: int) -> list[MessageList]:
        raise NotImplementedError

    def get_expected_prompt_length(self) -> int:
        raise NotImplementedError

    def get_expected_prefix_length(self) -> int:
        raise NotImplementedError

    def get_expected_output_length(self) -> int:
        raise NotImplementedError

    def get_max_output_length(self) -> int:
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
            create_message_list_from_prompt(prompt) for prompt in self.create_prompts(n)
        ]

    def get_expected_prompt_length(self) -> int:
        return self.prompt_length

    def get_expected_output_length(self) -> int:
        return self.output_length

    def get_expected_prefix_length(self) -> int:
        return self.prefix_length

    def get_max_output_length(self) -> int:
        return self.get_expected_output_length()


class PromptBasedLoad(LoadType, frozen=True):
    type: Literal["prompts"] = "prompts"
    format: Literal["chat", "text"] = "chat"
    sources: list[str]
    inject_tokens: bool = False
    inject_after_str: str = ""
    inject_after_occurrence: Literal["first", "last"] = "last"
    prompt_filler_tokens: AtLeastTwoUniqueStrings = DEFAULT_PROMPT_FILLER_TOKENS
    _prompts: list[MessageList] = PrivateAttr()
    _prompt_sampler: Generator[MessageList, None, None] = PrivateAttr()
    _filler_sampler: Generator[str, None, None] = PrivateAttr()
    expected_prompt_length: int = 0
    expected_output_length: int = 0
    expected_prefix_length: int = 0
    max_output_length: int = 0

    def model_post_init(self, context: Any, /) -> None:
        object.__setattr__(self, "_prompts", read_prompts(self.sources, self.format))
        object.__setattr__(self, "_prompt_sampler", sample_exhaustively(self._prompts))
        object.__setattr__(
            self,
            "_filler_sampler",
            sample_unique_concatenations_exhaustively(self.prompt_filler_tokens),
        )

    def create_message_lists(self, n: int) -> list[MessageList]:
        message_lists = []
        for _ in range(n):
            prompt = next(self._prompt_sampler)
            prompt = split_off_last_assistant_answer(prompt)[0]
            if self.inject_tokens:
                fill_tokens = next(self._filler_sampler)
                prompt = inject_into_prompt(
                    prompt,
                    self.inject_after_str,
                    self.inject_after_occurrence,
                    fill_tokens,
                )
            message_lists.append(prompt)
        return message_lists

    def get_expected_prompt_length(self) -> int:
        return self.expected_prompt_length

    def get_expected_output_length(self) -> int:
        return self.expected_output_length

    def get_expected_prefix_length(self) -> int:
        return self.expected_prefix_length

    def get_max_output_length(self) -> int:
        return self.max_output_length


SpecificLoadType = Annotated[
    HeuristicLoad | PromptBasedLoad, Field(discriminator="type")
]
