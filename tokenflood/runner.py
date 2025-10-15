import asyncio
from typing import List

import numpy as np
from litellm import acompletion
from litellm.types.utils import ModelResponse
from tqdm import tqdm
from tokenflood.models.endpoint_spec import EndpointSpec

def create_schedule(requests_per_second: float, test_length_in_seconds: float) -> List[float]:
    """Create a randomized schedule with length guarantees."""
    if requests_per_second <= 0:
        raise ValueError("requests per second cannot be zero or negative")
    if test_length_in_seconds < 0:
        raise ValueError("test length in seconds cannot be negative")
    num_requests = int(requests_per_second * test_length_in_seconds)
    if num_requests == 0:
        raise ValueError("Total number of requests would be zero")
    pauses = np.random.exponential(1/requests_per_second, size=num_requests)
    total_length = pauses.sum()
    pauses = pauses / (total_length / test_length_in_seconds)
    return list(pauses)


async def run_test(schedule: List[float], prompts: List[str], num_generation_tokens_list: List[int], endpoint_spec: EndpointSpec) -> List[ModelResponse]:
    requests: List[asyncio.Task] = []
    for pause, prompt, num_generation_tokens in tqdm(zip(schedule, prompts, num_generation_tokens_list)):
        requests.append(
            asyncio.create_task(send_llm_request(endpoint_spec, prompt, num_generation_tokens))
        )
        await asyncio.sleep(pause)

    responses: List[ModelResponse] = await asyncio.gather(*requests)
    return responses

async def send_llm_request(endpoint_spec: EndpointSpec, prompt: str, num_generation_tokens: int) -> ModelResponse:
    messages = [{"content": prompt, "role": "user"}]
    return await acompletion(model=endpoint_spec.model, messages=messages, max_tokens=num_generation_tokens,
                             base_url=endpoint_spec.base_url, api_key=endpoint_spec.api_key_env_var,
                             deployment_id=endpoint_spec.deployment, extra_headers=endpoint_spec.extra_headers)
