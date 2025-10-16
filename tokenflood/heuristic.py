from typing import List, Tuple

from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.messages import MessageList, create_message_list_from_prompt
from tokenflood.models.test_spec import HeuristicTestSpec
from tokenflood.models.token_set import TokenSet

heuristic_tasks = [
    HeuristicTask(
        task="Ignore the random input and continue the following sequence up to 10000 without abbreviation: 1 2 3 4"
    )
]

heuristic_token_sets = [
    TokenSet(tokens=[" " + chr(c) for c in range(65, 91)]),  # " A" - " Z"
]


def create_prompt_prefix(token_set: TokenSet, length: int) -> str:
    """Create a predictable prefix for prompts."""
    return token_set.tokens[0] * max(0, length)


def create_prompt_random_part(token_set: TokenSet, length: int) -> str:
    return "".join(token_set.sample(length))


def create_prompt(
    token_set: TokenSet, prompt_length: int, prefix_length: int, task: HeuristicTask
) -> str:
    task_tokens = task.roughly_estimated_token_cost
    random_prompt_tokens = prompt_length - prefix_length - task_tokens
    prompt = create_prompt_prefix(token_set, prefix_length)
    if random_prompt_tokens > 0:
        prompt += create_prompt_random_part(token_set, random_prompt_tokens)
    prompt += task.task
    return prompt


def create_heuristic_messages(
    heuristic_test_spec: HeuristicTestSpec,
) -> List[Tuple[MessageList, int]]:
    task = heuristic_tasks[0]
    token_set = heuristic_token_sets[0]
    messages = []
    for prompt_length, prefix_length, output_length in zip(
        heuristic_test_spec.sample()
    ):
        prompt = create_prompt(token_set, prompt_length, prefix_length, task)
        messages.append((create_message_list_from_prompt(prompt), output_length))
    return messages
