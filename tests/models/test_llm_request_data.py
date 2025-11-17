import logging

import pytest

from tokenflood.models.llm_request_data import LLMRequestData
from tokenflood.util import get_exact_date_str


@pytest.fixture()
def default_llm_request_data():
    return LLMRequestData(
        datetime=get_exact_date_str(),
        requests_per_second_phase=1.0,
        group_id="1",
        request_number=1,
        model="hf/standard",
        latency=100,
        expected_input_tokens=1000,
        measured_input_tokens=1000,
        expected_prefix_tokens=500,
        measured_prefix_tokens=500,
        expected_output_tokens=32,
        measured_output_tokens=32,
        generated_text="1 2 3 4",
        prompt="Count up to 10000.",
    )


@pytest.mark.parametrize(
    "data_update, should_warn, warning_content",
    [
        ({}, False, ""),
        ({"measured_input_tokens": 1200}, True, "input tokens that are 20% longer"),
        ({"measured_input_tokens": 700}, True, "input tokens that are 30% shorter"),
        ({"measured_output_tokens": 64}, True, "output tokens that are 100% longer"),
        ({"measured_output_tokens": 16}, True, "output tokens that are 50% shorter"),
    ],
)
def test_warn_on_diverging_measurements(
    default_llm_request_data, data_update, should_warn, warning_content, caplog
):
    llm_request_data = default_llm_request_data.model_copy(update=data_update)
    with caplog.at_level(logging.WARNING):
        llm_request_data.warn_on_diverging_measurements()
    if should_warn:
        assert warning_content in caplog.text
    else:
        assert caplog.text == ""
