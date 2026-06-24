import asyncio
import logging

from tokenflood.io import IOContext
from tokenflood.logging_utils import global_warn_once_filter
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.data.error_data import ErrorContext
from tokenflood.models.data.llm_request_data import LLMRequestContext
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.models.data.ping_request_data import PingRequestContext
from tokenflood.networking import (
    ObserveURLMiddleware,
    option_request_endpoint,
    time_async_func,
)
from tokenflood.runner import (
    get_warm_session,
    handle_llm_result,
    handle_ping_result,
    send_llm_request,
)
from tokenflood.schedule import create_even_schedule
from tokenflood.util import get_exact_date_str

log = logging.getLogger(__name__)


async def run_observation(
    endpoint_spec: EndpointSpec,
    observation_spec: ObservationSpec,
    io_context: IOContext,
):
    url_observer = ObserveURLMiddleware()
    io_context.activate()
    await io_context.wait_for_pending_writes()
    log.info("Warming up.")
    error = await get_warm_session(endpoint_spec, io_context)

    if error:
        log.error(f"Not starting observation due to error: {error}")
        await io_context.wait_for_pending_writes()
        return

    llm_request_tasks = set()
    ping_tasks = set()
    num_pings = 0
    load_type = observation_spec.load_type
    message_lists = load_type.create_message_lists(observation_spec.total_num_requests)
    request_per_second_phase = observation_spec.requests_per_second_during_polling
    i = 0
    burst_pauses = create_even_schedule(
        observation_spec.num_requests, observation_spec.within_seconds
    )
    inter_polling_pause = observation_spec.get_inter_polling_pause()
    log.info(f"Doing {observation_spec.num_polls} polls in total.")
    for poll_idx in range(observation_spec.num_polls):
        log.info(f"Starting poll {poll_idx + 1}.")
        error_context = ErrorContext(
            requests_per_second_phase=request_per_second_phase, group_id=str(poll_idx)
        )
        for burst_idx in range(observation_spec.num_requests):
            request_context = LLMRequestContext(
                datetime=get_exact_date_str(),
                expected_input_tokens=load_type.get_expected_prompt_length(),
                expected_prefix_tokens=load_type.get_expected_prefix_length(),
                expected_output_tokens=load_type.get_expected_output_length(),
                requests_per_second_phase=request_per_second_phase,
                request_number=i,
                model=endpoint_spec.provider_model_str,
                prompt=message_lists[i][0]["content"],
                group_id=str(poll_idx),
            )
            log.info(
                f"starting request number {i + 1} (number {burst_idx + 1} within current poll)"
            )
            t = asyncio.create_task(
                send_llm_request(
                    endpoint_spec,
                    message_lists[i],
                    load_type.get_max_output_length(),
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
                await asyncio.sleep(burst_pauses[burst_idx])
            i += 1
        if poll_idx < observation_spec.num_polls - 1:
            log.info(f"Sleeping {inter_polling_pause}s until next polling phase")
            await asyncio.sleep(inter_polling_pause)
            global_warn_once_filter.clear()

    log.info("Waiting for all requests to come back.")
    while llm_request_tasks or ping_tasks:
        await asyncio.sleep(1.0)
    # make sure all data can be flushed
    await io_context.wait_for_pending_writes()
