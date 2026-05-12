import pytest

from tokenflood.messages import create_message_list_from_prompt
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.networking import (
    ObserveURLMiddleware,
    option_request_endpoint,
    ping_endpoint,
    time_async_func,
)
from tokenflood.runner import send_llm_request


@pytest.mark.asyncio
async def test_observe_url_middleware(
    base_endpoint_spec: EndpointSpec,
    url_observer: ObserveURLMiddleware,
    with_patched_aiohttp_session,
):
    prompt = "ping"
    messages = create_message_list_from_prompt(prompt)
    response = await send_llm_request(base_endpoint_spec, messages, 1)
    assert response
    assert url_observer.host == "127.0.0.1"
    assert url_observer.port == 8000
    assert url_observer.session is not None


@pytest.mark.asyncio
async def test_unpatched_observe_url_middleware(
    base_endpoint_spec: EndpointSpec, url_observer: ObserveURLMiddleware
):
    assert url_observer.host is None
    assert url_observer.port is None
    assert url_observer.session is None
    prompt = "ping"
    messages = create_message_list_from_prompt(prompt)
    response = await send_llm_request(base_endpoint_spec, messages, 1)
    assert response
    assert url_observer.host is None
    assert url_observer.port is None
    assert url_observer.session is None


@pytest.mark.asyncio
async def test_ping_endpoint():
    latency = await time_async_func(ping_endpoint("127.0.0.1", 8000))
    assert latency < 100


@pytest.mark.asyncio
async def test_option_request_endpoint(
    base_endpoint_spec: EndpointSpec,
    with_patched_aiohttp_session,
    url_observer: ObserveURLMiddleware,
):
    prompt = "ping"
    messages = create_message_list_from_prompt(prompt)
    await send_llm_request(base_endpoint_spec, messages, 1)
    latency = await time_async_func(
        option_request_endpoint(
            url_observer.session, str(url_observer.url), url_observer.headers
        )
    )
    assert latency < 100


@pytest.mark.asyncio
async def test_observe_url_middleware_singleton(
    base_endpoint_spec: EndpointSpec,
    url_observer: ObserveURLMiddleware,
    with_patched_aiohttp_session,
):
    prompt = "ping"
    messages = create_message_list_from_prompt(prompt)
    response = await send_llm_request(base_endpoint_spec, messages, 1)
    assert response
    assert url_observer.host == "127.0.0.1"
    assert url_observer.port == 8000
    assert url_observer.session is not None

    url_observer_2 = ObserveURLMiddleware()

    assert url_observer == url_observer_2

    assert url_observer.host == "127.0.0.1"
    assert url_observer.port == 8000
    assert url_observer.session is not None
