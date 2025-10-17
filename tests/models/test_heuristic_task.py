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


def test_heuristic_task_rough_token_estimation():
    task_description = "Please write a letter to Santa Claus"
    task = HeuristicTask(task=task_description)
    assert task.roughly_estimated_token_cost > 0


def test_get_token_cost(test_tokenizer: Tokenizer):
    task_description = "Please write a letter to Santa Claus"
    task = HeuristicTask(task=task_description)
    assert task.get_token_cost(test_tokenizer) == len(
        test_tokenizer.encode(task_description).tokens
    )
