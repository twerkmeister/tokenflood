import pytest
from aiohttp import ClientSession

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.networking import (
    ObserveURLMiddleware,
    option_request_endpoint,
    ping_endpoint,
    time_async_func,
)
from tokenflood.runner import send_llm_request


@pytest.mark.asyncio
async def test_observe_url_middleware(base_endpoint_spec: EndpointSpec):
    url_observer = ObserveURLMiddleware()
    client_session = ClientSession(middlewares=[url_observer])
    prompt = "ping"
    messages = [{"content": prompt, "role": "user"}]
    response = await send_llm_request(base_endpoint_spec, messages, 1, client_session)
    assert response
    assert url_observer.host == "127.0.0.1"
    assert url_observer.port == 8000


@pytest.mark.asyncio
async def test_ping_endpoint():
    latency = await time_async_func(ping_endpoint("127.0.0.1", 8000))
    assert latency < 5


@pytest.mark.asyncio
async def test_option_request_endpoint(base_endpoint_spec: EndpointSpec):
    url_observer = ObserveURLMiddleware()
    client_session = ClientSession(middlewares=[url_observer])
    prompt = "ping"
    messages = [{"content": prompt, "role": "user"}]
    await send_llm_request(base_endpoint_spec, messages, 1, client_session)
    latency = await time_async_func(
        option_request_endpoint(
            client_session, str(url_observer.url), url_observer.headers
        )
    )
    assert latency < 5
