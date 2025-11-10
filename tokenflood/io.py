import asyncio
import csv
import os
from collections import deque
from io import StringIO
from typing import Any, Callable, Dict, List, Type, TypeVar

import aiofiles
import yaml
from pydantic import BaseModel

from tokenflood.constants import (
    ERROR_RING_BUFFER_SIZE,
    RESULTS_FOLDER,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.error_data import ErrorData
from tokenflood.models.llm_request_data import LLMRequestData
from tokenflood.models.ping_request_data import PingData
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.util import get_fields

T = TypeVar("T", bound=BaseModel)


def read_pydantic_yaml(class_type: Type[T]) -> Callable[[str], T]:
    def read_class_type(filename: str) -> T:
        with open(filename) as f:
            data = yaml.safe_load(f)
        return class_type(**data)

    return read_class_type


def read_pydantic_yaml_list(class_type: Type[T]) -> Callable[[str], List[T]]:
    def read_class_type_list(filename: str) -> List[T]:
        with open(filename) as f:
            list_data = yaml.safe_load(f)
        return [class_type(**data) for data in list_data]

    return read_class_type_list


def write_pydantic_yaml(filename: str, o: T) -> None:
    with open(filename, "w") as f:
        yaml.safe_dump(o.model_dump(), f, sort_keys=False)


def write_pydantic_yaml_list(filename: str, objects: List[T]) -> None:
    with open(filename, "w") as f:
        yaml.safe_dump([o.model_dump() for o in objects], f, sort_keys=False)


def read_endpoint_spec(filename: str) -> EndpointSpec:
    return read_pydantic_yaml(EndpointSpec)(filename)


def read_run_suite(filename: str) -> HeuristicRunSuite:
    return read_pydantic_yaml(HeuristicRunSuite)(filename)


def make_run_folder(run_name: str) -> str:
    run_folder = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(run_folder)
    return run_folder


def add_suffix_to_file_name(filename: str, suffix: str) -> str:
    filename, ext = os.path.splitext(filename)
    return f"{filename}{suffix}{ext}"


def get_first_available_filename_like(filename: str) -> str:
    """Find the first version of a file name that isn't taken yet.

    Adds a suffix to find another available version."""
    suffix = 1
    current_name = filename
    while os.path.exists(current_name):
        current_name = add_suffix_to_file_name(filename, f"_{suffix:02d}")
        suffix += 1
    return current_name


def list_dir_relative(folder_name: str) -> List[str]:
    """List dir while preserving the relative path name."""
    return [os.path.join(folder_name, f) for f in os.listdir(folder_name)]


def write_file(filename: str, s: str):
    with open(filename, "w") as f:
        f.write(s)


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def error_to_str(e: BaseException) -> str:
    return str(e)


def exception_group_to_str(eg: ExceptionGroup) -> str:
    return "\n".join([error_to_str(e) for e in eg.exceptions])


class FileSink:
    def __init__(self, destination: str):
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.destination = destination
        self.consumer_task = None
        self.closed = False

    async def _consume(self):
        async with aiofiles.open(self.destination, "w", encoding="utf-8") as f:
            while not self.closed:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=2)
                    await f.write(item)
                    await f.flush()
                except asyncio.TimeoutError:
                    pass

    def write(self, item: str):
        self.queue.put_nowait(item)

    def close(self):
        self.closed = True

    def activate(self):
        self.consumer_task = asyncio.create_task(self._consume())

    async def wait_for_pending_writes(self):
        while not self.queue.empty():
            await asyncio.sleep(0.001)


class CSVFileSink(FileSink):
    def __init__(self, destination: str, columns: List[str]):
        super().__init__(destination)
        self.columns = columns
        self.stringio = StringIO()
        self.writer = csv.DictWriter(self.stringio, fieldnames=columns)
        self.write_header()

    def write_dict(self, item: Dict[str, Any]):
        self.writer.writerow(item)
        self.flush()

    def write_header(self):
        self.writer.writeheader()
        self.flush()

    def flush(self):
        self.write(self.get_buffer())
        self.reset_buffer()

    def get_buffer(self) -> str:
        self.stringio.seek(0)
        return self.stringio.read()

    def reset_buffer(self):
        self.stringio.seek(0)
        self.stringio.truncate(0)


class IOContext:
    def __init__(self):
        self.state_watch = deque(maxlen=ERROR_RING_BUFFER_SIZE)

    def error_rate(self) -> float:
        if len(self.state_watch) == 0:
            return 0.0
        return sum(self.state_watch) / len(self.state_watch)

    def write_error(self, data: Dict):
        raise NotImplementedError

    def write_llm_request(self, data: Dict):
        raise NotImplementedError

    def write_network_latency(self, data: Dict):
        raise NotImplementedError

    def activate(self):
        raise NotImplementedError

    def wait_for_pending_writes(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class FileIOContext(IOContext):
    def __init__(self, llm_request_file, network_latency_file, error_file):
        super().__init__()
        self.llm_request_sink = CSVFileSink(
            llm_request_file, columns=get_fields(LLMRequestData)
        )
        self.network_latency_sink = CSVFileSink(
            network_latency_file, columns=get_fields(PingData)
        )
        self.error_sink = CSVFileSink(error_file, columns=get_fields(ErrorData))

    def write_error(self, data: Dict):
        self.error_sink.write_dict(data)
        self.state_watch.append(1)

    def write_llm_request(self, data: Dict):
        self.llm_request_sink.write_dict(data)
        self.state_watch.append(0)

    def write_network_latency(self, data: Dict):
        self.network_latency_sink.write_dict(data)

    def activate(self):
        self.error_sink.activate()
        self.network_latency_sink.activate()
        self.llm_request_sink.activate()

    async def wait_for_pending_writes(self):
        await self.error_sink.wait_for_pending_writes()
        await self.llm_request_sink.wait_for_pending_writes()
        await self.network_latency_sink.wait_for_pending_writes()

    def close(self):
        self.error_sink.close()
        self.llm_request_sink.close()
        self.network_latency_sink.close()
