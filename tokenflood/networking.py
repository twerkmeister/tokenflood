import asyncio
import time
from asyncio import Protocol
from typing import Coroutine
from functools import wraps

from aiohttp import ClientHandlerType, ClientRequest, ClientResponse, ClientSession
from multidict import CIMultiDict

from tokenflood.constants import CLIENT_SESSION_INIT_BACKUP_ATTR


class ObserveURLMiddleware:
    _instance = None

    async def __call__(
        self, req: ClientRequest, handler: ClientHandlerType
    ) -> ClientResponse:
        self.url = req.url
        self.host = req.url.host
        self.port = req.url.port
        self.headers = req.headers
        self.session = req.session
        return await handler(req)

    @classmethod
    def reset(cls):
        if cls._instance:
            cls._instance.url = None
            cls._instance.host = None
            cls._instance.port = None
            cls._instance.headers = None
            cls._instance.session = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObserveURLMiddleware, cls).__new__(cls)
            cls._instance.url = None
            cls._instance.host = None
            cls._instance.port = None
            cls._instance.headers = None
            cls._instance.session = None
        return cls._instance


async def time_async_func(coroutine: Coroutine) -> int:
    start = time.time()
    await coroutine
    end = time.time()
    return int((end - start) * 1000)


async def ping_endpoint(host: str, port: int):
    """Ping endpoint and return latency in ms."""
    await asyncio.get_running_loop().create_connection(
        lambda: Protocol(), host=host, port=port
    )


async def option_request_endpoint(
    session: ClientSession, url: str, headers: CIMultiDict[str]
):
    if session is None or url is None or headers is None:
        raise ValueError(
            "Cannot send option request to endpoint without previously observed session, url or headers."
        )
    del_keys = [
        key
        for key in headers.keys()
        if key.startswith("x-stainless") or key == "content-length"
    ]
    for key in del_keys:
        del headers[key]
    await session.options(url, headers=headers)


def patch_aiohttp_client_session():
    if not hasattr(ClientSession, CLIENT_SESSION_INIT_BACKUP_ATTR):
        original_init = ClientSession.__init__

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            # Retrieve existing middlewares or start with an empty list
            middlewares = list(kwargs.get("middlewares", []))
            observe_url_middleware = ObserveURLMiddleware()
            # Check if your middleware is already present to avoid duplicates
            if observe_url_middleware not in middlewares:
                middlewares.append(observe_url_middleware)

            # Update kwargs with the new middleware list
            kwargs["middlewares"] = tuple(middlewares)

            # Call the original __init__ with the modified arguments
            original_init(self, *args, **kwargs)

        setattr(ClientSession, CLIENT_SESSION_INIT_BACKUP_ATTR, original_init)
        ClientSession.__init__ = patched_init


def unpatch_aiohttp_client_session():
    if hasattr(ClientSession, CLIENT_SESSION_INIT_BACKUP_ATTR):
        ClientSession.__init__ = getattr(ClientSession, CLIENT_SESSION_INIT_BACKUP_ATTR)
        delattr(ClientSession, CLIENT_SESSION_INIT_BACKUP_ATTR)
