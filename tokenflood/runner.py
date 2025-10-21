import asyncio
from typing import List, Optional

import numpy as np
from litellm import acompletion
from litellm.types.utils import ModelResponse, Usage
from tqdm import tqdm

from tokenflood.heuristic import (
    create_heuristic_messages,
    heuristic_tasks,
    heuristic_token_sets,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.messages import MessageList
from tokenflood.models.results import Results
from tokenflood.models.run_data import RunData
from tokenflood.models.run_spec import HeuristicRunSpec, RunSpec
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.token_set import TokenSet


def create_schedule(run_spec: RunSpec) -> List[float]:
    """Create a randomized schedule with a guaranteed total length."""
    pauses = np.random.exponential(
        1 / run_spec.requests_per_second, size=run_spec.total_num_requests
    )
    total_length = pauses.sum()
    pauses = pauses / (total_length / run_spec.test_length_in_seconds)
    return list(pauses)


def collect_results(
    message_lists: List[MessageList],
    expected_input_lengths: List[int],
    expected_prefix_lengths: List[int],
    expected_output_lengths: List[int],
    model_responses: List[ModelResponse],
) -> Results:
    usages: List[Usage] = [mr.usage for mr in model_responses]  # type: ignore[attr-defined]
    return Results(
        prompts=[ml[0]["content"] for ml in message_lists],
        generated_texts=[mr.choices[0]["message"]["content"] for mr in model_responses],
        latencies=tuple([int(mr._response_ms) for mr in model_responses]),  # type: ignore[attr-defined]
        expected_input_lengths=tuple(expected_input_lengths),
        expected_prefix_lengths=tuple(expected_prefix_lengths),
        expected_output_lengths=tuple(expected_output_lengths),
        measured_input_lengths=tuple([usage.prompt_tokens for usage in usages]),
        measured_prefix_lengths=tuple(
            [
                usage.prompt_tokens_details.cached_tokens or 0
                if usage.prompt_tokens_details
                else 0
                for usage in usages
            ]
        ),
        measured_output_lengths=tuple([usage.completion_tokens for usage in usages]),
    )


async def run_heuristic_test(
    run_spec: HeuristicRunSpec,
    endpoint_spec: EndpointSpec,
    token_set: Optional[TokenSet] = None,
    task: Optional[HeuristicTask] = None,
) -> RunData:
    token_set = token_set or heuristic_token_sets[0]
    task = task or heuristic_tasks[0]
    schedule = create_schedule(run_spec)
    prompt_lengths, prefix_lengths, output_lengths = run_spec.sample()
    message_lists = create_heuristic_messages(
        prompt_lengths, prefix_lengths, token_set, task
    )
    model_responses = await run_test(
        schedule, message_lists, output_lengths, endpoint_spec
    )
    results = collect_results(
        message_lists, prompt_lengths, prefix_lengths, output_lengths, model_responses
    )
    return RunData(run_spec=run_spec, responses=model_responses, results=results)


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


async def run_suite(
    endpoint_spec: EndpointSpec, suite: HeuristicRunSuite
) -> List[RunData]:
    run_specs = suite.create_run_specs()
    run_suite_data = []
    for run_spec in run_specs:
        run_data = await run_heuristic_test(run_spec, endpoint_spec)
        run_suite_data.append(run_data)
    return run_suite_data
