import os.path
import time

import numpy as np
import pytest

from tokenflood.io import write_pydantic_yaml_list
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
async def test_run_entire_tiny_suite(
    tiny_run_suite, base_endpoint_spec, tiny_run_data_file_unsafe
):
    run_suite_data = await run_suite(base_endpoint_spec, tiny_run_suite)
    run_specs = tiny_run_suite.create_run_specs()
    assert len(run_suite_data) == len(run_specs)
    for i in range(len(run_specs)):
        assert len(run_suite_data[i].responses) == run_specs[i].total_num_requests

    # writing it out if it doesn't exist
    if not os.path.exists(tiny_run_data_file_unsafe):
        write_pydantic_yaml_list(tiny_run_data_file_unsafe, run_suite_data)
