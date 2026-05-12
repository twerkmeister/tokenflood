import asyncio
import logging
import os
from unittest import mock

import pandas as pd
import pytest

from tokenflood.models.load_types.load_type import HeuristicLoad
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.observer import run_observation
from tokenflood.schedule import create_even_schedule


@pytest.fixture
def default_observation_spec():
    return ObservationSpec(
        name="test",
        duration_hours=24,
        polling_interval_minutes=20,
        load_type=HeuristicLoad(
            prompt_length=1024, prefix_length=512, output_length=20
        ),
        num_requests=4,
        within_seconds=1.0,
    )


@pytest.mark.parametrize(
    "num_requests, within_seconds, expected_result",
    [
        (5, 1.0, [0.25, 0.25, 0.25, 0.25, 0.0]),
        (4, 1.5, [0.5, 0.5, 0.5, 0.0]),
        (1, 2.0, [0.0]),
    ],
)
def test_create_even_schedule(num_requests, within_seconds, expected_result):
    assert expected_result == create_even_schedule(num_requests, within_seconds)


@pytest.mark.asyncio
async def test_run_observation(
    superfast_observation_spec,
    base_endpoint_spec,
    file_io_context,
    with_patched_aiohttp_session,
    url_observer,
):
    await run_observation(
        base_endpoint_spec, superfast_observation_spec, file_io_context
    )

    error_df = pd.read_csv(file_io_context.error_sink.destination)
    assert len(error_df) == 0

    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == superfast_observation_spec.total_num_requests

    ping_df = pd.read_csv(file_io_context.network_latency_sink.destination)
    assert len(ping_df) == superfast_observation_spec.num_polls


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
async def test_run_observation_openai_missing_api_key(
    superfast_observation_spec,
    openai_endpoint_spec,
    file_io_context,
    caplog,
    with_patched_aiohttp_session,
    url_observer,
):
    with caplog.at_level(logging.ERROR):
        await run_observation(
            openai_endpoint_spec, superfast_observation_spec, file_io_context
        )
    assert "API key" in caplog.text

    await asyncio.sleep(0.1)

    df = pd.read_csv(file_io_context.llm_request_sink.destination)
    assert len(df) == 0

    ping_df = pd.read_csv(file_io_context.network_latency_sink.destination)
    assert len(ping_df) == 0

    error_df = pd.read_csv(file_io_context.error_sink.destination)
    assert len(error_df) == 1
