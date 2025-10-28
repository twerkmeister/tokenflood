from typing import List

from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.messages import MessageList, create_message_list_from_prompt
from tokenflood.models.token_set import TokenSet

heuristic_tasks = [
    HeuristicTask(
        task="Task: Count up to 10000 naming each individual number like this: 1 2 3 4"
    ),
    HeuristicTask(
        task="Task: Write an intro chapter for a historic novel that is at least 20 pages long."
    ),
    HeuristicTask(
        task="Task: Recite the Bible verse by verse starting with Genesis 1."
    ),
]

heuristic_token_sets = [
    TokenSet(tokens=tuple([" " + chr(c) for c in range(65, 91)])),  # " A" - " Z"
]


def create_prompt_prefix(token_set: TokenSet, length: int) -> str:
    """Create a predictable prefix for prompts."""
    return token_set.tokens[0] * max(0, length)


def create_prompt_random_part(token_set: TokenSet, length: int) -> str:
    """Create the random prompt part between prefix and task."""
    return "".join(token_set.sample(length))


def create_prompt(
    token_set: TokenSet, prompt_length: int, prefix_length: int, task: HeuristicTask
) -> str:
    """Create a prompt that suffices the length constraints.

    The prompt will is structured like this:

    1. Common Prefix (using the TokenSet)
    2. Random Tokens (using the TokenSet)
    3. A single newline to separate the task from the random part
    4. The Task
    """
    task_tokens = task.roughly_estimated_token_cost
    random_prompt_tokens = prompt_length - prefix_length - task_tokens
    prompt = create_prompt_prefix(token_set, prefix_length)
    if random_prompt_tokens > 0:
        prompt += create_prompt_random_part(token_set, random_prompt_tokens)
    prompt += "\n"
    prompt += task.task
    return prompt


def create_heuristic_messages(
    prompt_lengths: List[int],
    prefix_lengths: List[int],
    token_set: TokenSet,
    task: HeuristicTask,
) -> List[MessageList]:
    messages = []
    for i in range(len(prompt_lengths)):
        prompt = create_prompt(token_set, prompt_lengths[i], prefix_lengths[i], task)
        messages.append(create_message_list_from_prompt(prompt))
    return messages
