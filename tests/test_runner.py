import time

import numpy as np
import pytest

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
async def test_send_llm_request(base_endpoint_spec: EndpointSpec):
    prompt = "Write down the ABC."
    messages = [{"content": prompt, "role": "user"}]
    response = await send_llm_request(base_endpoint_spec, messages, 10)
    assert response


@pytest.mark.asyncio
async def test_run_heuristic_test(run_spec, base_endpoint_spec):
    start = time.time()
    run_data = await run_heuristic_test(run_spec, base_endpoint_spec)
    end = time.time()
    assert end - start < run_spec.test_length_in_seconds + 5
    assert len(run_data.responses) == run_spec.total_num_requests
    assert len(run_data.results.prompts) == run_spec.total_num_requests


@pytest.mark.asyncio
async def test_run_entire_suite(tiny_run_suite, base_endpoint_spec):
    run_suite_data = await run_suite(base_endpoint_spec, tiny_run_suite)
    assert len(run_suite_data) == 2
    assert len(run_suite_data[0].responses) == 2
    assert len(run_suite_data[1].responses) == 6
