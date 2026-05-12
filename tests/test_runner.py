import asyncio
import logging
import os.path
import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tokenflood.messages import create_message_list_from_prompt
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_specs.load_test_spec import LoadTestPhase
from tokenflood.runner import (
    get_warm_session,
    make_empty_response,
    run_load_test_phase,
    run_load_test,
    send_llm_request,
)
from tokenflood.schedule import create_load_test_phase_schedule


@pytest.mark.parametrize(
    "requests_per_second, test_length_in_seconds", [(3, 10), (1, 5), (2, 400)]
)
def test_create_schedule(requests_per_second: float, test_length_in_seconds: int):
    run_spec = LoadTestPhase(
        requests_per_second=requests_per_second,
        duration_seconds=test_length_in_seconds,
    )
    schedule = create_load_test_phase_schedule(run_spec, 1)
    assert np.allclose(sum(schedule), test_length_in_seconds)
    assert len(schedule) == int(requests_per_second * test_length_in_seconds)


@pytest.mark.asyncio
async def test_send_llm_request(base_endpoint_spec: EndpointSpec):
    prompt = "ping"
    messages = create_message_list_from_prompt(prompt)
    response = await send_llm_request(base_endpoint_spec, messages, 1)
    assert response


@pytest.mark.asyncio
async def test_run_load_test_phase(
    tiny_load_test_spec,
    base_endpoint_spec,
    file_io_context,
    with_patched_aiohttp_session,
):
    file_io_context.activate()
    error = await get_warm_session(base_endpoint_spec, file_io_context)
    load_test_phase = tiny_load_test_spec.create_load_test_phases()[0]
    assert error is None
    start = time.time()
    error_threshold_tripped = await run_load_test_phase(
        "test",
        0,
        tiny_load_test_spec,
        load_test_phase,
        base_endpoint_spec,
        file_io_context,
    )
    end = time.time()
    assert end - start < load_test_phase.duration_seconds + 5
    assert not error_threshold_tripped
    await asyncio.sleep(0.1)
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == load_test_phase.total_num_requests


@pytest.mark.asyncio
async def test_run_entire_tiny_load_test(
    tiny_load_test_spec,
    base_endpoint_spec,
    file_io_context,
    with_patched_aiohttp_session,
):
    await run_load_test(base_endpoint_spec, tiny_load_test_spec, file_io_context)
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    total_num_requests = sum(
        [
            load_phase.total_num_requests
            for load_phase in tiny_load_test_spec.create_load_test_phases()
        ]
    )
    assert len(df) == total_num_requests


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
async def test_run_tiny_suite_openai_missing_api_key(
    tiny_load_test_spec,
    openai_endpoint_spec,
    file_io_context,
    caplog,
    with_patched_aiohttp_session,
):
    with caplog.at_level(logging.ERROR):
        await run_load_test(openai_endpoint_spec, tiny_load_test_spec, file_io_context)
    await asyncio.sleep(0.1)
    run_specs = tiny_load_test_spec.create_load_test_phases()
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == 0
    assert len(run_specs) > 1
    assert "API key" in caplog.text


@pytest.mark.asyncio
async def test_run_tiny_suite_bad_endpoint(
    tiny_load_test_spec,
    base_endpoint_spec,
    file_io_context,
    caplog,
    with_patched_aiohttp_session,
):
    # creating endpoint spec with bad port number
    bad_endpoint_spec = base_endpoint_spec.model_copy(
        update={"base_url": "http://127.0.0.1:8001/v1"}
    )
    with caplog.at_level(logging.ERROR):
        await run_load_test(bad_endpoint_spec, tiny_load_test_spec, file_io_context)
    await asyncio.sleep(0.1)
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == 0
    assert len(tiny_load_test_spec.create_load_test_phases()) > 1
    assert "Cannot connect" in caplog.text


@mock.patch("tokenflood.runner.warm_up_session")
@pytest.mark.asyncio
async def test_run_tiny_suite_bad_endpoint_but_fake_warmup(
    mocked_warm_up, tiny_load_test_spec, base_endpoint_spec, file_io_context, caplog
):
    mocked_warm_up.return_value = make_empty_response()
    # creating endpoint spec with bad port number
    bad_endpoint_spec = base_endpoint_spec.model_copy(
        update={"base_url": "http://127.0.0.1:8001/v1"}
    )
    with caplog.at_level(logging.ERROR):
        await run_load_test(bad_endpoint_spec, tiny_load_test_spec, file_io_context)
    assert len(caplog.messages) == 4
    assert caplog.messages[0].endswith("[Connect call failed ('127.0.0.1', 8001)]")
    assert caplog.messages[1].endswith(
        "observed session, url or headers."
    ) or caplog.messages[1].endswith("Session is closed")
    assert caplog.messages[2].startswith("Aborting the phase")
    assert caplog.messages[3].startswith("Ending the run because")
