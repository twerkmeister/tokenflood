import pytest

from tokenflood.models.load_types.load_type import HeuristicLoad


@pytest.fixture()
def heuristic_load() -> HeuristicLoad:
    return HeuristicLoad(prompt_length=1024, prefix_length=128, output_length=12)


@pytest.mark.parametrize(
    "num_prefix_tokens, expected_result", [(4, " A" * 4), (0, ""), (-10, "")]
)
def test_create_prompt_prefix(
    num_prefix_tokens: int, expected_result: str, heuristic_load
):
    assert heuristic_load.create_prompt_prefix(num_prefix_tokens) == expected_result


def test_create_prompt_random_part(heuristic_load):
    p1 = heuristic_load.create_prompt_random_part(1000)
    p2 = heuristic_load.create_prompt_random_part(1000)
    assert p1 != p2
    assert len(p1) == len(p2)


def test_create_prompt(heuristic_load):
    prompt = heuristic_load.create_prompt()

    assert len(prompt) > 2048
    assert (
        prompt[: heuristic_load.prefix_length * 2]
        == heuristic_load.prompt_filler_tokens[0] * heuristic_load.prefix_length
    )
    assert (
        prompt[heuristic_load.prefix_length * 2 : heuristic_load.prefix_length * 2 + 32]
        != heuristic_load.prompt_filler_tokens[0] * 16
    )
    assert prompt.endswith(heuristic_load.task)


def test_create_heuristic_messages(heuristic_load, tokenizer):
    message_lists = heuristic_load.create_message_lists(10)

    tokenized = tokenizer.encode_batch([m[0]["content"] for m in message_lists])

    # lengths should be within +-10 token range of the desired length
    assert all([1014 <= len(t.tokens) <= 1034 for t in tokenized])
