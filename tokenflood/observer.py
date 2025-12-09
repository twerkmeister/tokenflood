import asyncio
import logging
from typing import List

import numpy as np

from tokenflood.heuristic import create_heuristic_messages
from tokenflood.io import IOContext
from tokenflood.logging import global_warn_once_filter
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.error_data import ErrorContext
from tokenflood.models.llm_request_data import LLMRequestContext
from tokenflood.models.observation_spec import ObservationSpec
from tokenflood.models.ping_request_data import PingRequestContext
from tokenflood.networking import option_request_endpoint, time_async_func
from tokenflood.runner import (
    get_warm_session,
    handle_llm_result,
    handle_ping_result,
    send_llm_request,
)
from tokenflood.util import get_exact_date_str

log = logging.getLogger(__name__)


def create_even_schedule(num_requests: int, within_seconds: float) -> List[float]:
    if num_requests <= 1:
        return []
    return list(np.diff(np.linspace(0, within_seconds, num_requests)))


def create_schedule(observation_spec: ObservationSpec) -> List[float]:
    burst_pauses = create_even_schedule(
        observation_spec.num_requests, observation_spec.within_seconds
    )
    inter_polling_pause = observation_spec.polling_interval_minutes * 60 - sum(
        burst_pauses
    )
    section = [round(pause, 2) for pause in burst_pauses + [inter_polling_pause]]
    return section * observation_spec.num_polls()


async def run_observation(
    endpoint_spec: EndpointSpec,
    observation_spec: ObservationSpec,
    io_context: IOContext,
):
    io_context.activate()
    await io_context.wait_for_pending_writes()
    log.info("Warming up.")
    client_session, url_observer, error = await get_warm_session(
        endpoint_spec, io_context
    )

    if error:
        log.error(f"Not starting observation due to error: {error}")
        await client_session.close()
        return

    llm_request_tasks = set()
    ping_tasks = set()
    num_pings = 0
    prompt_lengths = [
        observation_spec.load_type.prompt_length
    ] * observation_spec.total_num_requests()
    prefix_lengths = [
        observation_spec.load_type.prefix_length
    ] * observation_spec.total_num_requests()
    output_lengths = [
        observation_spec.load_type.output_length
    ] * observation_spec.total_num_requests()
    message_lists = create_heuristic_messages(
        prompt_lengths,
        prefix_lengths,
        observation_spec.token_set,
        observation_spec.task,
    )
    request_per_second_phase = (
        observation_spec.num_requests / observation_spec.within_seconds
    )
    i = 0
    burst_pauses = create_even_schedule(
        observation_spec.num_requests, observation_spec.within_seconds
    )
    inter_polling_pause = (
        observation_spec.polling_interval_minutes * 60 - observation_spec.within_seconds
    )
    log.info(f"Doing {observation_spec.num_polls()} polls in total.")
    for poll_idx in range(observation_spec.num_polls()):
        error_context = ErrorContext(
            requests_per_second_phase=request_per_second_phase, group_id=str(poll_idx)
        )
        for burst_idx in range(observation_spec.num_requests):
            request_context = LLMRequestContext(
                datetime=get_exact_date_str(),
                expected_input_tokens=prompt_lengths[i],
                expected_prefix_tokens=prefix_lengths[i],
                expected_output_tokens=output_lengths[i],
                requests_per_second_phase=request_per_second_phase,
                request_number=i,
                model=endpoint_spec.provider_model_str,
                prompt=message_lists[i][0]["content"],
                group_id=str(poll_idx),
            )
            log.info(f"starting request {i}")
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

            # ping once per poll
            if poll_idx >= num_pings:
                ping_context = PingRequestContext(
                    datetime=get_exact_date_str(),
                    endpoint_url=str(url_observer.url),
                    requests_per_second_phase=request_per_second_phase,
                    group_id=str(poll_idx),
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
            if len(burst_pauses) > 0:
                await asyncio.sleep(burst_pauses[0])
            i += 1
        if poll_idx < observation_spec.num_polls() - 1:
            log.info(f"Sleeping {inter_polling_pause}s until next polling phase")
            await asyncio.sleep(inter_polling_pause)
            global_warn_once_filter.clear()

    log.info("Waiting for all requests to come back.")
    while llm_request_tasks or ping_tasks:
        await asyncio.sleep(1.0)

    # make sure all data can be flushed
    await io_context.wait_for_pending_writes()
