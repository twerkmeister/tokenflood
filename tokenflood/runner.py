import asyncio
from typing import Callable, List, Optional, Tuple
import logging

import litellm
import numpy as np
from aiohttp import ClientSession, TCPConnector
from litellm import acompletion
from litellm.types.utils import ModelResponse, Usage
from tqdm import tqdm

from tokenflood.constants import ERROR_RATE_LIMIT, ERROR_RING_BUFFER_SIZE
from tokenflood.heuristic import (
    create_heuristic_messages,
    heuristic_tasks,
    heuristic_token_sets,
)
from tokenflood.io import IOContext, exception_group_to_str
from tokenflood.logging import global_warn_once_filter
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.error_data import ErrorContext, ErrorData
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


def handle_error(
    io_context: IOContext, error_context: ErrorContext
) -> Callable[[asyncio.Task], None]:
    """Callback to handle task errors."""

    def on_done(task: asyncio.Task):
        error = task.exception()
        if error is not None:
            io_context.write_error(
                ErrorData(
                    datetime=get_exact_date_str(),
                    type=type(error).__name__,
                    message=str(error),
                    request_per_second_phase=error_context.requests_per_second_phase,
                ).model_dump()
            )

    return on_done


def handle_llm_result(
    io_context: IOContext,
    llm_request_context: LLMRequestContext,
    error_context: ErrorContext,
) -> Callable[[asyncio.Task[ModelResponse]], None]:
    """Callback to handle llm request results and errors."""

    def on_done(task: asyncio.Task[ModelResponse]):
        handle_error(io_context, error_context)(task)
        if not task.cancelled() and not task.exception():
            model_response: ModelResponse = task.result()
            data = LLMRequestData.from_response_and_context(
                model_response, llm_request_context
            )
            data.warn_on_diverging_measurements()
            io_context.write_llm_request(data.model_dump())

    return on_done


def handle_ping_result(
    io_context: IOContext, ping_context: PingRequestContext, error_context: ErrorContext
) -> Callable[[asyncio.Task[int]], None]:
    """Callback to handle ping request results and errors."""

    def on_done(task: asyncio.Task[int]):
        handle_error(io_context, error_context)(task)
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
) -> bool:
    token_set = token_set or heuristic_token_sets[0]
    task = task or heuristic_tasks[0]
    schedule = create_schedule(run_spec)

    prompt_lengths, prefix_lengths, output_lengths = run_spec.sample()
    message_lists = create_heuristic_messages(
        prompt_lengths, prefix_lengths, token_set, task
    )
    error_context = ErrorContext(requests_per_second_phase=run_spec.requests_per_second)
    error_threshold_tripped = False
    error_rate = 0.0
    num_pings = 0
    llm_request_tasks = set()
    ping_tasks = set()

    pbar = tqdm(range(len(schedule)), desc=test_description)
    for i in pbar:
        error_rate = io_context.error_rate()
        pbar.set_postfix({"error rate": round(error_rate, 2)})
        if error_rate > ERROR_RATE_LIMIT:
            error_threshold_tripped = True
            break
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
        t = asyncio.create_task(
            send_llm_request(
                endpoint_spec,
                message_lists[i],
                output_lengths[i],
                client_session,
            )
        )
        t.add_done_callback(
            handle_llm_result(io_context, request_context, error_context)
        )
        llm_request_tasks.add(t)
        t.add_done_callback(llm_request_tasks.discard)

        await asyncio.sleep(schedule[i])
        # ping at most every second
        if sum(schedule[: i + 1]) > num_pings:
            ping_context = PingRequestContext(
                datetime=get_exact_date_str(),
                endpoint_url=str(url_observer.url),
                requests_per_second_phase=run_spec.requests_per_second,
            )
            pt = asyncio.create_task(
                time_async_func(
                    option_request_endpoint(
                        client_session,
                        str(url_observer.url),
                        url_observer.headers,
                    )
                )
            )
            ping_tasks.add(pt)
            pt.add_done_callback(
                handle_ping_result(io_context, ping_context, error_context)
            )
            pt.add_done_callback(ping_tasks.discard)
            num_pings += 1
    log.info("Waiting for all requests to come back.")
    while llm_request_tasks or ping_tasks:
        await asyncio.sleep(1.0)

    # make sure all data can be flushed
    await io_context.wait_for_pending_writes()
    if error_threshold_tripped:
        log.error(
            f"Aborting the phase because the error rate exceeded {int(error_rate * 100)}% for the last {ERROR_RING_BUFFER_SIZE} requests."
        )
    else:
        log.info("Finished the phase.")
    return error_threshold_tripped


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
    connector = TCPConnector(limit=1000, loop=asyncio.get_running_loop())
    client_session = ClientSession(middlewares=[url_observer], connector=connector)
    error_context = ErrorContext(requests_per_second_phase=-1.0)
    try:
        async with asyncio.TaskGroup() as tg:
            t = tg.create_task(warm_up_session(endpoint_spec, client_session))
            t.add_done_callback(handle_error(io_context, error_context))
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
        error_threshold_tripped = await run_heuristic_test(
            test_description,
            run_spec,
            endpoint_spec,
            client_session,
            url_observer,
            io_context,
        )

        if error_threshold_tripped:
            log.error("Ending the run because the error threshold was tripped.")
            break
        global_warn_once_filter.clear()
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
    proceed: bool,
) -> bool:
    estimated_input_tokens, estimated_output_tokens = estimate_token_usage(suite)
    log.info("Checking estimated token usage for the run:")
    input_token_color = get_limit_color(
        estimated_input_tokens, suite.input_token_budget
    )
    output_token_color = get_limit_color(
        estimated_output_tokens, suite.output_token_budget
    )
    log.info(
        f"Estimated input tokens / configured max input tokens: "
        f"[{input_token_color}]{estimated_input_tokens}[/] / [blue]{suite.input_token_budget}[/]"
    )
    log.info(
        f"Estimated output tokens / configured max output tokens: "
        f"[{output_token_color}]{estimated_output_tokens}[/] / [blue]{suite.output_token_budget}[/]"
    )
    if (
        estimated_input_tokens > suite.input_token_budget
        or estimated_output_tokens > suite.output_token_budget
    ):
        log.info("[red]Estimated tokens beyond configured budget. Aborting the run.[/]")
        log.info(
            "Increase the maximum tokens you are willing to spend by setting the variables "
            "[red]input_token_budget[/] and [red]output_token_budget[/] to a higher value "
            "in your run suite file."
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
