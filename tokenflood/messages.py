import os
import uuid
from functools import partial

from litellm import acount_tokens, token_counter
from tokenizers import Tokenizer

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.message_list import MessageList
from tokenflood.util import exec_sync_or_async_func

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


async def get_input_output_prefix_token_lengths(
    message_lists: list[MessageList],
    endpoint_spec: EndpointSpec,
    use_hf_tokenizer: bool,
) -> tuple[list[int], list[int], list[int], MessageList]:
    input_message_lists, output_message_lists = zip(
        *[split_off_last_assistant_answer(messages) for messages in message_lists]
    )
    common_prefix = get_common_prefix(input_message_lists)
    common_prefix_lengths = []
    valid_output_message_lists = [om for om in output_message_lists if om is not None]
    if use_hf_tokenizer:
        tokenizer = {
            "type": "huggingface_tokenizer",
            "tokenizer": Tokenizer.from_pretrained(endpoint_spec.model),
        }
        func = partial(
            token_counter, model=endpoint_spec.model, custom_tokenizer=tokenizer
        )
    else:
        func = partial(count_tokens_using_api, endpoint_spec=endpoint_spec)

    input_token_lengths = [
        await exec_sync_or_async_func(func, messages=m) for m in input_message_lists
    ]
    output_token_lengths = [
        await exec_sync_or_async_func(func, messages=m)
        for m in valid_output_message_lists
    ]
    if common_prefix:
        common_prefix_lengths = [
            await exec_sync_or_async_func(func, messages=common_prefix)
        ]

    return (
        input_token_lengths,
        output_token_lengths,
        common_prefix_lengths,
        common_prefix,
    )
