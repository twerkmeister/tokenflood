import asyncio
from typing import List, Optional

import numpy as np
from litellm import acompletion
from litellm.types.utils import ModelResponse
from tqdm import tqdm

from tokenflood.heuristic import (
    create_heuristic_messages,
    heuristic_tasks,
    heuristic_token_sets,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.messages import MessageList
from tokenflood.models.run_spec import HeuristicRunSpec, RunSpec
from tokenflood.models.token_set import TokenSet


def create_schedule(run_spec: RunSpec) -> List[float]:
    """Create a randomized schedule with a guaranteed total length."""
    pauses = np.random.exponential(
        1 / run_spec.requests_per_second, size=run_spec.total_num_requests
    )
    total_length = pauses.sum()
    pauses = pauses / (total_length / run_spec.test_length_in_seconds)
    return list(pauses)


async def run_heuristic_test(
    run_spec: HeuristicRunSpec,
    endpoint_spec: EndpointSpec,
    token_set: Optional[TokenSet] = None,
    task: Optional[HeuristicTask] = None,
) -> List[ModelResponse]:
    token_set = token_set or heuristic_token_sets[0]
    task = task or heuristic_tasks[0]
    schedule = create_schedule(run_spec)
    message_lists = create_heuristic_messages(run_spec, token_set, task)
    num_generation_tokens = run_spec.sample_output_token_counts()
    return await run_test(schedule, message_lists, num_generation_tokens, endpoint_spec)


async def run_test(
    schedule: List[float],
    message_lists: List[MessageList],
    num_generation_tokens: List[int],
    endpoint_spec: EndpointSpec,
) -> List[ModelResponse]:
    request_tasks: List[asyncio.Task] = []
    for i in tqdm(range(len(schedule))):
        request_task = asyncio.create_task(
            send_llm_request(endpoint_spec, message_lists[i], num_generation_tokens[i])
        )
        request_tasks.append(request_task)
        await asyncio.sleep(schedule[i])

    responses: List[ModelResponse] = await asyncio.gather(*request_tasks)
    return responses


async def send_llm_request(
    endpoint_spec: EndpointSpec, messages: MessageList, num_generation_tokens: int
) -> ModelResponse:
    return await acompletion(
        model=endpoint_spec.model,
        messages=messages,
        max_tokens=num_generation_tokens,
        base_url=endpoint_spec.base_url,
        api_key=endpoint_spec.api_key_env_var,
        deployment_id=endpoint_spec.deployment,
        extra_headers=endpoint_spec.extra_headers,
    )
