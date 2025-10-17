from typing import ContextManager
import pytest
from tokenizers import Tokenizer

from tests.utils import does_not_raise
from tokenflood.models.heuristic_task import HeuristicTask


@pytest.mark.parametrize(
    "task, expectation",
    [
        ("Please write a letter to Santa Claus", does_not_raise()),
        ("", pytest.raises(ValueError)),
    ],
)
def test_heuristic_task_validation(task: str, expectation: ContextManager):
    with expectation:
        HeuristicTask(task=task)


def test_heuristic_task_rough_token_estimation(heuristic_task):
    assert heuristic_task.roughly_estimated_token_cost > 0


def test_get_token_cost(tokenizer: Tokenizer, heuristic_task):
    assert heuristic_task.get_token_cost(tokenizer) == len(
        tokenizer.encode(heuristic_task.task).tokens
    )
