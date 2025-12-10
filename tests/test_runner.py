import logging
import os.path
import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from aiohttp import ClientSession

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_spec import RunSpec
from tokenflood.runner import (
    create_bursty_schedule,
    get_warm_session,
    make_empty_response,
    run_heuristic_test,
    run_suite,
    send_llm_request,
)


@pytest.mark.parametrize(
    "requests_per_second, test_length_in_seconds", [(3, 10), (1, 5), (2, 400)]
)
def test_create_schedule(requests_per_second: float, test_length_in_seconds: int):
    run_spec = RunSpec(
        requests_per_second=requests_per_second,
        test_length_in_seconds=test_length_in_seconds,
    )
    schedule = create_bursty_schedule(run_spec)
    assert np.allclose(sum(schedule), test_length_in_seconds)
    assert len(schedule) == int(requests_per_second * test_length_in_seconds)


@pytest.mark.asyncio
async def test_send_llm_request(base_endpoint_spec: EndpointSpec):
    client_session = ClientSession()
    prompt = "ping"
    messages = [{"content": prompt, "role": "user"}]
    response = await send_llm_request(base_endpoint_spec, messages, 1, client_session)
    assert response


@pytest.mark.asyncio
async def test_run_heuristic_test(
    base_run_suite, run_spec, base_endpoint_spec, file_io_context
):
    file_io_context.activate()
    client_session, url_observer, error = await get_warm_session(
        base_endpoint_spec, file_io_context
    )
    assert error is None
    start = time.time()
    error_threshold_tripped = await run_heuristic_test(
        "test",
        0,
        base_run_suite,
        run_spec,
        base_endpoint_spec,
        client_session,
        url_observer,
        file_io_context,
    )
    end = time.time()
    assert end - start < run_spec.test_length_in_seconds + 5
    assert not error_threshold_tripped

    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    # TODO: FLAKY
    # FAILED tests/test_runner.py::test_run_heuristic_test - assert 1 == 2
    assert len(df) == run_spec.total_num_requests


@pytest.mark.asyncio
async def test_run_entire_tiny_suite(
    tiny_run_suite,
    base_endpoint_spec,
    file_io_context,
):
    await run_suite(base_endpoint_spec, tiny_run_suite, file_io_context)
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    total_num_requests = sum(
        [run_spec.total_num_requests for run_spec in tiny_run_suite.create_run_specs()]
    )
    assert len(df) == total_num_requests


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
async def test_run_tiny_suite_openai_missing_api_key(
    tiny_run_suite, openai_endpoint_spec, file_io_context, caplog
):
    with caplog.at_level(logging.ERROR):
        await run_suite(openai_endpoint_spec, tiny_run_suite, file_io_context)
    run_specs = tiny_run_suite.create_run_specs()
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == 0
    assert len(run_specs) > 1
    assert "API key" in caplog.text


@pytest.mark.asyncio
async def test_run_tiny_suite_bad_endpoint(
    tiny_run_suite, base_endpoint_spec, file_io_context, caplog
):
    # creating endpoint spec with bad port number
    bad_endpoint_spec = base_endpoint_spec.model_copy(
        update={"base_url": "http://127.0.0.1:8001/v1"}
    )
    with caplog.at_level(logging.ERROR):
        await run_suite(bad_endpoint_spec, tiny_run_suite, file_io_context)
    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == 0
    assert len(tiny_run_suite.create_run_specs()) > 1
    assert "Connection error" in caplog.text


@mock.patch("tokenflood.runner.warm_up_session")
@pytest.mark.asyncio
async def test_run_tiny_suite_bad_endpoint_but_fake_warmup(
    mocked_warm_up, tiny_run_suite, base_endpoint_spec, file_io_context, caplog
):
    mocked_warm_up.return_value = make_empty_response()
    # creating endpoint spec with bad port number
    bad_endpoint_spec = base_endpoint_spec.model_copy(
        update={"base_url": "http://127.0.0.1:8001/v1"}
    )
    with caplog.at_level(logging.ERROR):
        await run_suite(bad_endpoint_spec, tiny_run_suite, file_io_context)
    assert len(caplog.messages) == 2
    assert caplog.messages[0].startswith("Aborting the phase")
    assert caplog.messages[1].startswith("Ending the run because")
