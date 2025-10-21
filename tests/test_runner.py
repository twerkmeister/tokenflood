import os
import time

import numpy as np
import pytest
from unittest import mock

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_spec import RunSpec
from tokenflood.runner import (
    create_schedule,
    run_heuristic_test,
    run_suite,
    send_llm_request,
)


@pytest.mark.parametrize(
    "requests_per_second, test_length_in_seconds", [(3, 10), (1, 5), (2, 400)]
)
def test_create_schedule(requests_per_second: float, test_length_in_seconds: int):
    run_spec = RunSpec(
        name="abc",
        requests_per_second=requests_per_second,
        test_length_in_seconds=test_length_in_seconds,
    )
    schedule = create_schedule(run_spec)
    assert np.allclose(sum(schedule), test_length_in_seconds)
    assert len(schedule) == int(requests_per_second * test_length_in_seconds)


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
async def test_send_llm_request(base_endpoint_spec: EndpointSpec):
    prompt = "Write down the ABC."
    messages = [{"content": prompt, "role": "user"}]
    response = await send_llm_request(base_endpoint_spec, messages, 10)
    assert response


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
async def test_run_heuristic_test(run_spec, base_endpoint_spec):
    start = time.time()
    responses, results = await run_heuristic_test(run_spec, base_endpoint_spec)
    end = time.time()
    assert end - start < run_spec.test_length_in_seconds + 5
    assert len(responses) == run_spec.total_num_requests
    assert len(results.prompts) == run_spec.total_num_requests


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
async def test_run_entire_suite(tiny_run_suite, base_endpoint_spec):
    suite_results = await run_suite(base_endpoint_spec, tiny_run_suite)
    assert len(suite_results) == 2
    assert len(suite_results[0][1]) == 2
    assert len(suite_results[1][1]) == 6
