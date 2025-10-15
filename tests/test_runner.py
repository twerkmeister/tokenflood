import numpy as np
import pytest

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.runner import create_schedule, send_llm_request


@pytest.mark.parametrize("requests_per_second, test_length_in_seconds",
                         [(3, 10), (1, 5), (2, 400)])
def test_create_schedule(requests_per_second: float, test_length_in_seconds: float):
    schedule = create_schedule(requests_per_second, test_length_in_seconds)
    assert np.allclose(sum(schedule), test_length_in_seconds)
    assert len(schedule) == int(requests_per_second * test_length_in_seconds)

@pytest.mark.parametrize("requests_per_second, test_length_in_seconds",
                         [(-3, 10), (1, -5), (0, 4), (4, 0), (0.1, 1)])
def test_create_schedule_edge_cases(requests_per_second: float, test_length_in_seconds: float):
    with pytest.raises(ValueError):
        create_schedule(requests_per_second, test_length_in_seconds)

@pytest.mark.asyncio
async def test_send_llm_request():
    prompt = "Write down the ABC."
    endpoint_spec = EndpointSpec(model="openai/HuggingFaceTB/SmolLM-135M-Instruct", base_url="http://127.0.0.1:8000/v1")
    response = await send_llm_request(endpoint_spec, prompt, 10)
    assert response
