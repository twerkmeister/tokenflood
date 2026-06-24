import copy
import os
import random
import uuid
import bisect
from functools import partial
from typing import Literal

from litellm import acount_tokens, token_counter
from tokenizers import Tokenizer

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.message_list import MessageList

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"


def make_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def make_user_message(content: str) -> dict[str, str]:
    return make_message(USER_ROLE, content)


def make_assistant_message(content: str) -> dict[str, str]:
    return make_message(ASSISTANT_ROLE, content)


def create_message_list_from_prompt(prompt: str) -> MessageList:
    return [make_user_message(prompt)]


# unique strings that won't appear in the prompts
MESSAGE_SEPERATOR = uuid.uuid4().hex
ROLE_CONTENT_SEPERATOR = uuid.uuid4().hex


def apply_fake_chat_template(messages: MessageList) -> str:
    return MESSAGE_SEPERATOR.join(
        [f"{m['role']}{ROLE_CONTENT_SEPERATOR}{m['content']}" for m in messages]
    )


def simulate_prefix_caching(sorted_input_strings: list[str]) -> list[str]:
    """Simulates the process of prefix caching and returns the best prefix for each incoming query."""
    if not len(sorted_input_strings):
        return []

    best_prefixes: list[str] = []
    seen_indices: list[int] = []
    random_order = list(range(len(sorted_input_strings)))
    random.shuffle(random_order)
    for idx in random_order:
        if len(seen_indices) == 0:
            seen_indices.append(idx)
            best_prefixes.append("")
        else:
            insertion_pos = bisect.bisect_left(seen_indices, idx)
            previous_idx = (
                seen_indices[insertion_pos - 1] if insertion_pos > 0 else None
            )
            # pre insertion, the next idx is still at insertion pos
            next_idx = (
                seen_indices[insertion_pos]
                if insertion_pos < len(seen_indices)
                else None
            )

            prefix_with_previous_idx = ""
            prefix_with_next_idx = ""
            if previous_idx is not None:
                prefix_with_previous_idx = os.path.commonprefix(
                    [sorted_input_strings[previous_idx], sorted_input_strings[idx]]
                )
            if next_idx is not None:
                prefix_with_next_idx = os.path.commonprefix(
                    [sorted_input_strings[idx], sorted_input_strings[next_idx]]
                )
            best_prefix = (
                prefix_with_previous_idx
                if len(prefix_with_previous_idx) > len(prefix_with_next_idx)
                else prefix_with_next_idx
            )

            best_prefixes.append(best_prefix)
            seen_indices.insert(insertion_pos, idx)
    return best_prefixes


def parse_fake_chat_template(s: str) -> MessageList:
    if s == "":
        return []
    message_parts = s.split(MESSAGE_SEPERATOR)
    messages = []
    for part in message_parts:
        if not part.strip():
            continue
        role, content = part.split(ROLE_CONTENT_SEPERATOR)
        messages.append(make_message(role, content))
    return messages


def split_off_last_assistant_answer(
    messages: MessageList,
) -> tuple[MessageList, MessageList | None]:
    if len(messages) > 0 and messages[-1]["role"] == ASSISTANT_ROLE:
        return messages[:-1], messages[-1:]
    else:
        return messages, None


def inject_into_prompt(
    messages: MessageList,
    inject_after_str: str,
    inject_after_occurrence: Literal["last", "first"],
    s: str,
) -> MessageList:
    messages = copy.deepcopy(messages)
    step = 1 if inject_after_occurrence == "first" else -1
    for message in messages[::step]:
        if inject_after_str in message["content"]:
            if inject_after_occurrence == "first":
                pre, post = message["content"].split(inject_after_str, 1)
            else:
                pre, post = message["content"].rsplit(inject_after_str, 1)

            message["content"] = pre + inject_after_str + s + post
            break
    return messages


def get_common_prefix(input_message_lists: list[MessageList]) -> MessageList:
    if len(input_message_lists) == 0:
        return []
    input_strings = [
        apply_fake_chat_template(messages) for messages in input_message_lists
    ]
    sorted_input_strings = sorted(input_strings)
    # first and the last one are the most distinct strings in terms of prefixes
    prefix = os.path.commonprefix([sorted_input_strings[0], sorted_input_strings[-1]])
    return parse_fake_chat_template(prefix)


def get_prefixes_from_simulation(
    input_message_lists: list[MessageList],
) -> list[MessageList]:
    if len(input_message_lists) == 0:
        return []
    input_strings = [
        apply_fake_chat_template(messages) for messages in input_message_lists
    ]
    sorted_input_strings = sorted(input_strings)
    best_prefixes = simulate_prefix_caching(sorted_input_strings)
    # first and the last one are the most distinct strings in terms of prefixes
    return [parse_fake_chat_template(prefix) for prefix in best_prefixes]


async def count_tokens_using_api(
    messages: MessageList, endpoint_spec: EndpointSpec
) -> int:
    response = await acount_tokens(
        model=endpoint_spec.provider_model_str,
        messages=messages,
        api_key=os.getenv(endpoint_spec.api_key_env_var)
        if endpoint_spec.api_key_env_var
        else None,
        api_base=endpoint_spec.base_url,
    )
    return response.total_tokens


async def count_tokens_using_tokenizer(
    messages: MessageList, model: str, tokenizer: dict
):
    return token_counter(model, custom_tokenizer=tokenizer, messages=messages)


async def get_input_output_prefix_token_lengths(
    message_lists: list[MessageList],
    endpoint_spec: EndpointSpec | None,
    hf_tokenizer: str | None,
) -> tuple[list[int], list[int], list[int], list[int], MessageList]:
    if len(message_lists) == 0:
        return [], [], [], [], []
    input_message_lists, output_message_lists = [], []
    for messages in message_lists:
        input_messages, output_messages = split_off_last_assistant_answer(messages)
        input_message_lists.append(input_messages)
        if output_messages is not None:
            output_message_lists.append(output_messages)

    common_prefix = get_common_prefix(input_message_lists)
    prefixes_from_simulation = get_prefixes_from_simulation(input_message_lists)
    common_prefix_lengths = []
    simulation_prefix_lengths = []
    if hf_tokenizer is not None:
        tokenizer = {
            "type": "huggingface_tokenizer",
            "tokenizer": Tokenizer.from_pretrained(hf_tokenizer),
        }
        func = partial(
            count_tokens_using_tokenizer, tokenizer=tokenizer, model=hf_tokenizer
        )
    elif endpoint_spec is not None:
        func = partial(count_tokens_using_api, endpoint_spec=endpoint_spec)
    else:
        raise ValueError("Either tokenizer or endpoint must be defined.")
    input_token_lengths = [await func(messages=m) for m in input_message_lists]
    output_token_lengths = [await func(messages=m) for m in output_message_lists]
    if common_prefix:
        common_prefix_lengths = [await func(messages=common_prefix)]
    if prefixes_from_simulation:
        simulation_prefix_lengths = [
            await func(messages=m) for m in prefixes_from_simulation
        ]

    return (
        input_token_lengths,
        output_token_lengths,
        common_prefix_lengths,
        simulation_prefix_lengths,
        common_prefix,
    )
