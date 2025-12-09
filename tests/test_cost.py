import pytest

from tokenflood.cost import check_token_usage_upfront
from tokenflood.models.budget import Budget


@pytest.mark.parametrize(
    "user_input, input_token_diff, output_token_diff, autoaccept, expected_result",
    [
        # within limits no auto accept
        ("y", 100, 100, False, True),
        ("yes", 100, 100, False, True),
        ("n", 100, 100, False, False),
        ("no", 100, 100, False, False),
        ("gibberish", 100, 100, False, False),
        # within limits auto accept
        ("gibberish", 100, 100, True, True),
        # out of limits no auto accept
        ("gibberish", -100, 100, False, False),
        ("gibberish", 100, -10, False, False),
        # out of limits auto accept
        ("gibberish", -100, 100, True, False),
        ("gibberish", 100, -10, True, False),
    ],
)
def test_check_token_usage_upfront(
    tiny_run_suite,
    monkeypatch,
    user_input,
    input_token_diff,
    output_token_diff,
    autoaccept,
    expected_result,
):
    monkeypatch.setattr("builtins.input", lambda _: user_input)

    estimated_input_tokens, estimated_output_tokens = (
        tiny_run_suite.get_input_output_token_cost()
    )

    update = {
        "budget": Budget(
            input_tokens=estimated_input_tokens + input_token_diff,
            output_tokens=estimated_output_tokens + output_token_diff,
        )
    }

    tiny_run_suite = tiny_run_suite.model_copy(update=update)

    assert expected_result == check_token_usage_upfront(
        tiny_run_suite,
        tiny_run_suite.budget,
        autoaccept,
    )
