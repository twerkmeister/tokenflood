import pytest

from tokenflood.heuristic import (
    create_prompt,
    create_prompt_prefix,
    create_prompt_random_part,
)
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.token_set import TokenSet


@pytest.mark.parametrize(
    "num_prefix_tokens, expected_result", [(4, " A" * 4), (0, ""), (-10, "")]
)
def test_create_prompt_prefix(num_prefix_tokens: int, expected_result: str):
    token_set = TokenSet(tokens=[" A", " B"])
    assert create_prompt_prefix(token_set, num_prefix_tokens) == expected_result


def test_create_prompt_random_part():
    token_set = TokenSet(tokens=[" A", " B", " C", " D"])

    p1 = create_prompt_random_part(token_set, 1000)
    p2 = create_prompt_random_part(token_set, 1000)
    assert p1 != p2
    assert len(p1) == len(p2)


def test_create_prompt():
    token_set = TokenSet(tokens=[" A", " B", " C", " D", " E"])
    task = HeuristicTask(task="--- Write a letter to Santa Claus")
    prompt_length = 1024
    prefix_length = 128
    prompt = create_prompt(token_set, prompt_length, prefix_length, task)

    assert len(prompt) > 2048
    assert prompt[: prefix_length * 2] == token_set.tokens[0] * prefix_length
    assert (
        prompt[prefix_length * 2 : prefix_length * 2 + 32] != token_set.tokens[0] * 16
    )
    assert prompt.endswith(task.task)
