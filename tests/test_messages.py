import pytest

from tokenflood.messages import (
    apply_fake_chat_template,
    ROLE_CONTENT_SEPERATOR,
    MESSAGE_SEPERATOR,
    parse_fake_chat_template,
    USER_ROLE,
    make_user_message,
    make_assistant_message,
    ASSISTANT_ROLE,
    split_off_last_assistant_answer,
    get_common_prefix,
    count_tokens_using_api,
    get_input_output_prefix_token_lengths,
    get_prefixes_from_simulation,
)


@pytest.mark.parametrize(
    "messages, expected_messages_in_template",
    [
        (
            [make_user_message("a"), make_assistant_message("b")],
            f"{USER_ROLE}{ROLE_CONTENT_SEPERATOR}a{MESSAGE_SEPERATOR}{ASSISTANT_ROLE}{ROLE_CONTENT_SEPERATOR}b",
        ),
        ([make_user_message("a")], f"{USER_ROLE}{ROLE_CONTENT_SEPERATOR}a"),
        ([], ""),
    ],
)
def test_apply_fake_chat_template(messages, expected_messages_in_template):
    messages_in_template = apply_fake_chat_template(messages)
    assert messages_in_template == expected_messages_in_template


@pytest.mark.parametrize(
    "messages_in_template,expected_messages",
    [
        (
            f"{USER_ROLE}{ROLE_CONTENT_SEPERATOR}a{MESSAGE_SEPERATOR}{ASSISTANT_ROLE}{ROLE_CONTENT_SEPERATOR}b",
            [make_user_message("a"), make_assistant_message("b")],
        ),
        (f"{USER_ROLE}{ROLE_CONTENT_SEPERATOR}", [make_user_message("")]),
        ("", []),
    ],
)
def test_parse_fake_chat_template(messages_in_template, expected_messages):
    messages = parse_fake_chat_template(messages_in_template)
    assert messages == expected_messages


@pytest.mark.parametrize(
    "messages,expected_head_idx,expected_tail_idx",
    [
        ([make_user_message("a")], 1, None),
        ([make_user_message("a"), make_assistant_message("b")], 1, 1),
        (
            [
                make_user_message("a"),
                make_assistant_message("b"),
                make_user_message("a"),
            ],
            3,
            None,
        ),
        ([], 0, None),
        ([make_assistant_message("a")], 0, 0),
    ],
)
def test_split_off_last_assistant_answer(
    messages, expected_head_idx, expected_tail_idx
):
    head, tail = split_off_last_assistant_answer(messages)
    assert head == messages[:expected_head_idx]
    assert tail == (
        [messages[expected_tail_idx]]
        if expected_tail_idx is not None
        else expected_tail_idx
    )


@pytest.mark.parametrize(
    "messages_list, expected_prefix",
    [
        (
            [[make_user_message("abc")], [make_user_message("def")]],
            [make_user_message("")],
        ),
        (
            [[make_assistant_message("abc")], [make_assistant_message("def")]],
            [make_assistant_message("")],
        ),
        ([[make_assistant_message("abc")], [make_user_message("abc")]], []),
        (
            [[make_user_message("abcdef")], [make_user_message("abcghi")]],
            [make_user_message("abc")],
        ),
        (
            [
                [
                    make_user_message("a"),
                    make_assistant_message("b"),
                    make_user_message("c"),
                ],
                [
                    make_user_message("a"),
                    make_assistant_message("b"),
                    make_user_message("d"),
                ],
            ],
            [
                make_user_message("a"),
                make_assistant_message("b"),
                make_user_message(""),
            ],
        ),
        (
            [
                [make_user_message("abc")],
                [make_user_message("abc")],
                [make_user_message("def")],
            ],
            [make_user_message("")],
        ),
    ],
)
def test_get_common_prefix(messages_list, expected_prefix):
    assert get_common_prefix(messages_list) == expected_prefix


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", [([make_user_message("abc")])])
async def test_count_tokens_using_api(messages, base_endpoint_spec):
    assert await count_tokens_using_api(messages, base_endpoint_spec) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_lists, tokenizer_name",
    [
        (
            [
                [make_user_message("abc")],
                [make_user_message("abcdef"), make_assistant_message("ghi")],
            ],
            "HuggingFaceTB/SmolLM-135M-Instruct",
        ),
        (
            [
                [make_user_message("abc")],
                [make_user_message("abcdef"), make_assistant_message("ghi")],
            ],
            None,
        ),
    ],
)
async def test_get_input_output_prefix_token_lengths(
    message_lists, tokenizer_name, base_endpoint_spec
):
    (
        input_lengths,
        output_lengths,
        prefix_length,
        simulated_prefix_lengths,
        _,
    ) = await get_input_output_prefix_token_lengths(
        message_lists, base_endpoint_spec, tokenizer_name
    )
    assert len(input_lengths) == 2
    assert len(output_lengths) == 1
    assert len(simulated_prefix_lengths) == 2
    assert len(prefix_length) == 1


@pytest.mark.parametrize(
    "message_lists, expected_prefixes",
    [
        (
            [
                [make_user_message("abc")],
                [make_user_message("abcdef"), make_assistant_message("ghi")],
            ],
            [[], [make_user_message("abc")]],
        ),
        (
            [
                [make_user_message("abc")],
                [make_user_message("def"), make_assistant_message("ghi")],
            ],
            [[], [make_user_message("")]],
        ),
    ],
)
def test_get_prefixes_from_simulation(message_lists, expected_prefixes):
    assert get_prefixes_from_simulation(message_lists) == expected_prefixes
