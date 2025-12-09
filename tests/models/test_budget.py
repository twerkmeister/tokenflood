import pytest

from tests.utils import does_not_raise
from tokenflood.constants import MAX_INPUT_TOKENS_DEFAULT, MAX_OUTPUT_TOKENS_DEFAULT
from tokenflood.models.budget import Budget


@pytest.mark.parametrize(
    "input_tokens, output_tokens, expectation",
    [
        (MAX_INPUT_TOKENS_DEFAULT, MAX_OUTPUT_TOKENS_DEFAULT, does_not_raise()),
        (0, MAX_OUTPUT_TOKENS_DEFAULT, pytest.raises(ValueError)),
        (MAX_INPUT_TOKENS_DEFAULT, 0, pytest.raises(ValueError)),
        (0, 0, pytest.raises(ValueError)),
    ],
)
def test_budget_validation(input_tokens, output_tokens, expectation):
    with expectation:
        Budget(input_tokens=input_tokens, output_tokens=output_tokens)
