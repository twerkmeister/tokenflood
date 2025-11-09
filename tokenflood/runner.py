import asyncio
from typing import Callable, List, Optional, Tuple
import logging

import litellm
import numpy as np
from aiohttp import ClientSession
from litellm import acompletion
from litellm.types.utils import ModelResponse, Usage
from tqdm import tqdm

from tokenflood.constants import MAX_INPUT_TOKENS_ENV_VAR, MAX_OUTPUT_TOKENS_ENV_VAR
from tokenflood.heuristic import (
    create_heuristic_messages,
    heuristic_tasks,
    heuristic_token_sets,
)
from tokenflood.io import IOContext, error_to_str, exception_group_to_str
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.llm_request_data import LLMRequestContext, LLMRequestData
from tokenflood.models.messages import MessageList, create_message_list_from_prompt
from tokenflood.models.ping_request_data import PingData, PingRequestContext
from tokenflood.models.run_spec import HeuristicRunSpec, RunSpec
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.token_set import TokenSet
from tokenflood.networking import (
    ObserveURLMiddleware,
    option_request_endpoint,
    time_async_func,
)
from tokenflood.util import get_exact_date_str

log = logging.getLogger(__name__)

litellm.disable_cache()


def handle_error(io_context: IOContext) -> Callable[[asyncio.Task], None]:
    """Callback to handle task errors."""

    def on_done(task: asyncio.Task):
        if task.cancelled():
            io_context.write_error("Request cancelled.")
        else:
            error = task.exception()
            if error is not None:
                io_context.write_error(error_to_str(error))

    return on_done


def handle_llm_result(
    io_context: IOContext, llm_request_context: LLMRequestContext
) -> Callable[[asyncio.Task[ModelResponse]], None]:
    """Callback to handle llm request results and errors."""

    def on_done(task: asyncio.Task[ModelResponse]):
        handle_error(io_context)(task)
        if not task.cancelled() and not task.exception():
            model_response: ModelResponse = task.result()
            data = LLMRequestData.from_response_and_context(
                model_response, llm_request_context
            )
            io_context.write_llm_request(data.model_dump())

    return on_done


def handle_ping_result(
    io_context: IOContext, ping_context: PingRequestContext
) -> Callable[[asyncio.Task[int]], None]:
    """Callback to handle ping request results and errors."""

    def on_done(task: asyncio.Task[int]):
        handle_error(io_context)(task)
        if not task.cancelled() and not task.exception():
            latency: int = task.result()
            data = PingData.from_context(ping_context, latency)
            io_context.write_network_latency(data.model_dump())

    return on_done


def make_empty_response() -> ModelResponse:
    """Create an empty ModelResponse object as a placeholder for skipped requests.

    Requests are skipped once an error happens during any requests."""
    return ModelResponse(
        choices=[{"message": {"content": ""}}],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        _response_ms=0,
    )


def create_schedule(run_spec: RunSpec) -> List[float]:
    """Create a randomized schedule with a guaranteed total length."""
    pauses = np.random.exponential(
        1 / run_spec.requests_per_second, size=run_spec.total_num_requests
    )
    total_length = pauses.sum()
    pauses = pauses / (total_length / run_spec.test_length_in_seconds)
    return list(pauses)


async def run_heuristic_test(
    test_description: str,
    run_spec: HeuristicRunSpec,
    endpoint_spec: EndpointSpec,
    client_session: ClientSession,
    url_observer: ObserveURLMiddleware,
    io_context: IOContext,
    token_set: Optional[TokenSet] = None,
    task: Optional[HeuristicTask] = None,
) -> Optional[str]:
    token_set = token_set or heuristic_token_sets[0]
    task = task or heuristic_tasks[0]
    schedule = create_schedule(run_spec)

    prompt_lengths, prefix_lengths, output_lengths = run_spec.sample()
    message_lists = create_heuristic_messages(
        prompt_lengths, prefix_lengths, token_set, task
    )
    error = None
    num_pings = 0
    try:
        async with asyncio.TaskGroup() as tg:
            for i in tqdm(range(len(schedule)), desc=test_description):
                request_context = LLMRequestContext(
                    datetime=get_exact_date_str(),
                    expected_input_tokens=prompt_lengths[i],
                    expected_prefix_tokens=prefix_lengths[i],
                    expected_output_tokens=output_lengths[i],
                    requests_per_second_phase=run_spec.requests_per_second,
                    request_number=i,
                    model=endpoint_spec.provider_model_str,
                    prompt=message_lists[i][0]["content"],
                )
                t = tg.create_task(
                    send_llm_request(
                        endpoint_spec,
                        message_lists[i],
                        output_lengths[i],
                        client_session,
                    )
                )
                t.add_done_callback(handle_llm_result(io_context, request_context))

                await asyncio.sleep(schedule[i])
                # ping at most every second
                if sum(schedule[: i + 1]) > num_pings:
                    ping_context = PingRequestContext(
                        datetime=get_exact_date_str(),
                        endpoint_url=str(url_observer.url),
                        requests_per_second_phase=run_spec.requests_per_second,
                    )
                    pt = tg.create_task(
                        time_async_func(
                            option_request_endpoint(
                                client_session,
                                str(url_observer.url),
                                url_observer.headers,
                            )
                        )
                    )
                    pt.add_done_callback(handle_ping_result(io_context, ping_context))
                    num_pings += 1
            log.info("Waiting for all requests to come back.")
    except ExceptionGroup as eg:
        error = exception_group_to_str(eg)
        log.error(f"Aborting the phase due to errors: {error}")

    # make sure all data can be flushed
    await io_context.wait_for_pending_writes()
    log.info("Finished the phase.")
    return error


async def warm_up_session(endpoint_spec: EndpointSpec, client_session: ClientSession):
    message_list = create_message_list_from_prompt("ping")
    return await send_llm_request(endpoint_spec, message_list, 1, client_session)


async def send_llm_request(
    endpoint_spec: EndpointSpec,
    messages: MessageList,
    num_generation_tokens: int,
    client_session: ClientSession,
) -> ModelResponse:
    return await acompletion(
        model=endpoint_spec.provider_model_str,
        messages=messages,
        max_tokens=num_generation_tokens,
        base_url=endpoint_spec.base_url,
        api_key=endpoint_spec.api_key_env_var,
        deployment_id=endpoint_spec.deployment,
        extra_headers=endpoint_spec.extra_headers,
        max_retries=0,
        shared_session=client_session,
    )


def make_test_description(
    suite: HeuristicRunSuite, phase: int, run_spec: HeuristicRunSpec
) -> str:
    return f"Run suite {suite.name} phase {phase}: {run_spec.requests_per_second:.2f} requests/s"


async def get_warm_session(
    endpoint_spec: EndpointSpec, io_context: IOContext
) -> Tuple[ClientSession, ObserveURLMiddleware, Optional[str]]:
    error = None
    url_observer = ObserveURLMiddleware()
    client_session = ClientSession(middlewares=[url_observer])
    try:
        async with asyncio.TaskGroup() as tg:
            t = tg.create_task(warm_up_session(endpoint_spec, client_session))
            t.add_done_callback(handle_error(io_context))
    except ExceptionGroup as eg:
        error = exception_group_to_str(eg)
    return client_session, url_observer, error


async def run_suite(
    endpoint_spec: EndpointSpec, suite: HeuristicRunSuite, io_context: IOContext
):
    io_context.activate()
    run_specs = suite.create_run_specs()
    log.info("Warming up.")
    client_session, url_observer, error = await get_warm_session(
        endpoint_spec, io_context
    )
    if error:
        log.error(f"Not starting run due to error: {error}")
        await client_session.close()
        return
    for phase, run_spec in enumerate(run_specs):
        test_description = make_test_description(suite, phase + 1, run_spec)
        error = await run_heuristic_test(
            test_description,
            run_spec,
            endpoint_spec,
            client_session,
            url_observer,
            io_context,
        )

        if error:
            log.error(f"Ending run due to error: {error}")
            break
    await client_session.close()


def estimate_token_usage(suite: HeuristicRunSuite) -> Tuple[int, int]:
    """Estimate total token usage based on the run suite parameters.

    Specifically: requests per seconds, length of test, load types."""
    total_input_tokens = 0
    total_output_tokens = 0
    for run_spec in suite.create_run_specs():
        input_tokens, _, output_tokens = run_spec.sample()
        total_input_tokens += sum(input_tokens)
        total_output_tokens += sum(output_tokens)
    return total_input_tokens, total_output_tokens


def check_token_usage_upfront(
    suite: HeuristicRunSuite,
    max_input_tokens: int,
    max_output_tokens: int,
    proceed: bool,
) -> bool:
    estimated_input_tokens, estimated_output_tokens = estimate_token_usage(suite)
    log.info("Checking estimated token usage for the run:")
    input_token_color = get_limit_color(estimated_input_tokens, max_input_tokens)
    output_token_color = get_limit_color(estimated_output_tokens, max_output_tokens)
    log.info(
        f"Estimated input tokens / configured max input tokens: "
        f"[{input_token_color}]{estimated_input_tokens}[/] / [blue]{max_input_tokens}[/]"
    )
    log.info(
        f"Estimated output tokens / configured max output tokens: "
        f"[{output_token_color}]{estimated_output_tokens}[/] / [blue]{max_output_tokens}[/]"
    )
    if (
        estimated_input_tokens > max_input_tokens
        or estimated_output_tokens > max_output_tokens
    ):
        log.info(
            "[red]Estimated tokens beyond configured maximum. Aborting the run.[/]"
        )
        log.info(
            "Increase the maximum tokens you are willing to spend via the env vars "
            f"[red]{MAX_INPUT_TOKENS_ENV_VAR}[/] and [red]{MAX_OUTPUT_TOKENS_ENV_VAR}[/]"
        )
        return False

    if proceed:
        log.info("Token usage [blue]auto-accepted[/blue]")
        return True

    response = "start_value"
    yes_answers = {"y", "yes"}
    no_answers = {"n", "no", ""}
    trials = 0
    while response not in yes_answers.union(no_answers) and trials < 3:
        response = input("Start the run? [y/N]: ")
        response = response.strip().lower()
        trials += 1
    return response in yes_answers


def get_limit_color(n: int, target: int) -> str:
    if n > target:
        return "red"
    return "blue"
