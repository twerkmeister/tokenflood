import asyncio
from typing import List

import numpy as np
from litellm import acompletion
from litellm.types.utils import ModelResponse
from tqdm import tqdm
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.messages import MessageList
from tokenflood.models.test_spec import TestSpec


def create_schedule(test_spec: TestSpec) -> List[float]:
    """Create a randomized schedule with a guaranteed total length."""
    pauses = np.random.exponential(1/test_spec.requests_per_second,
                                   size=test_spec.total_num_requests)
    total_length = pauses.sum()
    pauses = pauses / (total_length / test_spec.test_length_in_seconds)
    return list(pauses)

async def run_test(schedule: List[float], message_lists: List[MessageList], num_generation_tokens_list: List[int], endpoint_spec: EndpointSpec) -> List[ModelResponse]:
    requests: List[asyncio.Task] = []
    for pause, messages, num_generation_tokens in tqdm(zip(schedule, message_lists, num_generation_tokens_list)):
        requests.append(
            asyncio.create_task(send_llm_request(endpoint_spec, messages, num_generation_tokens))
        )
        await asyncio.sleep(pause)

    responses: List[ModelResponse] = await asyncio.gather(*requests)
    return responses

async def send_llm_request(endpoint_spec: EndpointSpec, messages: MessageList, num_generation_tokens: int) -> ModelResponse:
    return await acompletion(model=endpoint_spec.model, messages=messages, max_tokens=num_generation_tokens,
                             base_url=endpoint_spec.base_url, api_key=endpoint_spec.api_key_env_var,
                             deployment_id=endpoint_spec.deployment, extra_headers=endpoint_spec.extra_headers)
