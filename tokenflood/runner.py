import asyncio
import os
from typing import Callable, Optional
import logging

import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse, Usage
from tqdm import tqdm

from tokenflood.constants import ERROR_RING_BUFFER_SIZE
from tokenflood.io import IOContext
from tokenflood.logging import global_warn_once_filter
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.data.error_data import ErrorContext, ErrorData
from tokenflood.models.data.llm_request_data import LLMRequestContext, LLMRequestData
from tokenflood.models.messages import MessageList, create_message_list_from_prompt
from tokenflood.models.data.ping_request_data import PingData, PingRequestContext
from tokenflood.models.run_specs.load_spec import LoadSpec, LoadPhase
from tokenflood.networking import (
    ObserveURLMiddleware,
    option_request_endpoint,
    time_async_func,
)
from tokenflood.schedule import create_load_test_phase_schedule
from tokenflood.util import get_exact_date_str

log = logging.getLogger(__name__)

litellm.disable_cache()
litellm.suppress_debug_info = True


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
                    group_id=error_context.group_id,
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


async def run_heuristic_test(
    test_description: str,
    phase: int,
    load_spec: LoadSpec,
    load_phase: LoadPhase,
    endpoint_spec: EndpointSpec,
    io_context: IOContext,
) -> bool:
    schedule = create_load_test_phase_schedule(load_phase, load_spec.burstiness)
    load_type = load_spec.load_type
    message_lists = load_type.create_message_lists(len(schedule))
    error_context = ErrorContext(
        requests_per_second_phase=load_phase.requests_per_second, group_id=str(phase)
    )
    error_threshold_tripped = False
    error_rate = 0.0
    num_pings = 0
    llm_request_tasks = set()
    ping_tasks = set()
    url_observer = ObserveURLMiddleware()

    pbar = tqdm(range(len(schedule)), desc=test_description)
    for i in pbar:
        error_rate = io_context.error_rate()
        pbar.set_postfix({"error rate": round(error_rate, 2)})
        if error_rate > load_spec.error_limit:
            error_threshold_tripped = True
            break
        request_context = LLMRequestContext(
            datetime=get_exact_date_str(),
            expected_input_tokens=load_type.get_expected_prompt_length(),
            expected_prefix_tokens=load_type.get_expected_prefix_length(),
            expected_output_tokens=load_type.get_expected_output_length(),
            requests_per_second_phase=load_phase.requests_per_second,
            request_number=i,
            model=endpoint_spec.provider_model_str,
            prompt=message_lists[i][0]["content"],
            group_id=str(phase),
        )
        t = asyncio.create_task(
            send_llm_request(
                endpoint_spec,
                message_lists[i],
                load_type.get_expected_output_length(),
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
                requests_per_second_phase=load_phase.requests_per_second,
                group_id=str(phase),
            )
            pt = asyncio.create_task(
                time_async_func(
                    option_request_endpoint(
                        url_observer.session,
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


async def warm_up_session(endpoint_spec: EndpointSpec, idx: int = 0):
    message_list = create_message_list_from_prompt(f"warmup ping{idx}")
    return await send_llm_request(endpoint_spec, message_list, 20)


async def send_llm_request(
    endpoint_spec: EndpointSpec,
    messages: MessageList,
    num_generation_tokens: int,
) -> ModelResponse:
    try:
        response = await acompletion(
            model=endpoint_spec.provider_model_str,
            messages=messages,
            max_tokens=num_generation_tokens,
            base_url=endpoint_spec.base_url,
            api_key=os.getenv(endpoint_spec.api_key_env_var)
            if endpoint_spec.api_key_env_var is not None
            else None,
            deployment_id=endpoint_spec.deployment,
            extra_headers=endpoint_spec.extra_headers,
            extra_body=endpoint_spec.extra_body,
            max_retries=0,
            reasoning_effort=endpoint_spec.reasoning_effort,
        )
    except Exception as e:
        log.error(e)
        raise
    return response


def make_test_description(suite: LoadSpec, phase: int, run_spec: LoadPhase) -> str:
    return f"Load test {suite.name} phase {phase}: {run_spec.requests_per_second:.2f} requests/s"


async def get_warm_session(
    endpoint_spec: EndpointSpec, io_context: IOContext
) -> Optional[str]:
    error = None
    error_context = ErrorContext(requests_per_second_phase=-1.0, group_id="warmup")
    for i in range(20):
        t = asyncio.create_task(warm_up_session(endpoint_spec, i))
        t.add_done_callback(handle_error(io_context, error_context))
        result = (await asyncio.gather(t, return_exceptions=True))[0]
        if isinstance(result, Exception):
            error = str(result)
            break
    return error


async def run_load_test(
    endpoint_spec: EndpointSpec, load_spec: LoadSpec, io_context: IOContext
):
    io_context.activate()
    await io_context.wait_for_pending_writes()
    load_phases = load_spec.create_load_phases()
    log.info("Warming up.")
    error = await get_warm_session(endpoint_spec, io_context)
    if error:
        log.error(f"Not starting run due to error during warmup: {error}")
        # letting any writes finish
        await asyncio.sleep(0.1)
        return
    for phase, run_spec in enumerate(load_phases):
        test_description = make_test_description(load_spec, phase + 1, run_spec)
        error_threshold_tripped = await run_heuristic_test(
            test_description,
            phase,
            load_spec,
            run_spec,
            endpoint_spec,
            io_context,
        )
        if error_threshold_tripped:
            log.error("Ending the run because the error threshold was tripped.")
            break
        global_warn_once_filter.clear()
