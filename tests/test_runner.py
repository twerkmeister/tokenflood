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
    check_token_usage_upfront,
    create_schedule,
    estimate_token_usage,
    get_warm_session,
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
    schedule = create_schedule(run_spec)
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
async def test_run_heuristic_test(run_spec, base_endpoint_spec, file_io_context):
    file_io_context.activate()
    client_session, url_observer, state = await get_warm_session(
        base_endpoint_spec, file_io_context
    )
    assert state.error is None
    start = time.time()
    error = await run_heuristic_test(
        "test",
        run_spec,
        base_endpoint_spec,
        client_session,
        url_observer,
        file_io_context,
    )
    end = time.time()
    assert end - start < run_spec.test_length_in_seconds + 5
    assert error is None

    df = pd.read_csv(file_io_context.llm_request_sink.destination)
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


def test_estimate_token_usage_tiny(tiny_run_suite):
    estimated_input_tokens, estimated_output_tokens = estimate_token_usage(
        tiny_run_suite
    )
    num_requests = 8
    assert estimated_input_tokens == 256 * num_requests
    assert estimated_output_tokens == 2 * num_requests


def test_estimate_token_usage_base(base_run_suite):
    estimated_input_tokens, estimated_output_tokens = estimate_token_usage(
        base_run_suite
    )
    num_requests = 300
    input_tokens_diff = abs(
        estimated_input_tokens - np.average([1024, 1024, 1200]) * num_requests
    )
    # less than 1% diff
    assert input_tokens_diff / estimated_input_tokens < 0.01

    output_tokens_diff = abs(
        estimated_output_tokens - np.average([16, 16, 40]) * num_requests
    )
    assert output_tokens_diff / estimated_output_tokens < 0.05


@pytest.mark.parametrize(
    "user_input, input_token_diff, output_token_diff, autoaccept, expected_result",
    [
        # within limits no auto accept
        ("y", 100, 100, False, True),
        ("yes", 100, 100, False, True),
        ("n", 100, 100, False, False),
        ("no", 100, 100, False, False),
        ("gibberish", 100, 100, False, False),
        # within limits auto accept
        ("gibberish", 100, 100, True, True),
        # out of limits no auto accept
        ("gibberish", -100, 100, False, False),
        ("gibberish", 100, -100, False, False),
        # out of limits auto accept
        ("gibberish", -100, 100, True, False),
        ("gibberish", 100, -100, True, False),
    ],
)
def test_check_token_usage_upfront(
    tiny_run_suite,
    monkeypatch,
    user_input,
    input_token_diff,
    output_token_diff,
    autoaccept,
    expected_result,
):
    monkeypatch.setattr("builtins.input", lambda _: user_input)

    estimated_input_tokens, estimated_output_tokens = estimate_token_usage(
        tiny_run_suite
    )

    assert expected_result == check_token_usage_upfront(
        tiny_run_suite,
        estimated_input_tokens + input_token_diff,
        estimated_output_tokens + output_token_diff,
        autoaccept,
    )
