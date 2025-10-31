import asyncio
import time
from asyncio import Protocol
from typing import Dict

from aiohttp import ClientHandlerType, ClientRequest, ClientResponse, ClientSession


class ObserveURLMiddleware:
    def __init__(self):
        self.url = None
        self.host = None
        self.port = None
        self.headers = None

    async def __call__(
        self, req: ClientRequest, handler: ClientHandlerType
    ) -> ClientResponse:
        self.url = req.url
        self.host = req.url.host
        self.port = req.url.port
        self.headers = req.headers
        return await handler(req)


async def ping_endpoint(host: str, port: int) -> int:
    """Ping endpoint and return latency in ms."""
    start = time.time()
    await asyncio.get_running_loop().create_connection(
        lambda: Protocol(), host=host, port=port
    )
    end = time.time()
    return int((end - start) * 1000)


async def option_request_endpoint(
    session: ClientSession, url: str, headers: Dict
) -> int:
    start = time.time()
    del_keys = [
        key
        for key in headers.keys()
        if key.startswith("x-stainless") or key == "content-length"
    ]
    for key in del_keys:
        del headers[key]
    await session.options(url, headers=headers)
    end = time.time()
    return int((end - start) * 1000)
