import numpy as np
import pytest

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.test_spec import TestSpec
from tokenflood.runner import create_schedule, send_llm_request


@pytest.mark.parametrize(
    "requests_per_second, test_length_in_seconds", [(3, 10), (1, 5), (2, 400)]
)
def test_create_schedule(requests_per_second: float, test_length_in_seconds: float):
    test_spec = TestSpec(
        name="abc",
        requests_per_second=requests_per_second,
        test_length_in_seconds=test_length_in_seconds,
    )
    schedule = create_schedule(test_spec)
    assert np.allclose(sum(schedule), test_length_in_seconds)
    assert len(schedule) == int(requests_per_second * test_length_in_seconds)


@pytest.mark.asyncio
async def test_send_llm_request(test_endpoint_spec: EndpointSpec):
    prompt = "Write down the ABC."
    messages = [{"content": prompt, "role": "user"}]
    response = await send_llm_request(test_endpoint_spec, messages, 10)
    assert response
