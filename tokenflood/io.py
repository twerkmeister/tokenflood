import asyncio
import csv
import json
import os
from collections import deque
from io import StringIO
from typing import Any, Dict, List, Set, Literal

import aiofiles
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from tokenflood.constants import (
    COMMON_RESULT_FILES,
    ERROR_RING_BUFFER_SIZE,
    OBSERVATION_RESULT_FILES,
    RESULTS_FOLDER,
    LOAD_TEST_RESULT_FILES,
)
from tokenflood.messages import create_message_list_from_prompt
from tokenflood.models.data.error_data import ErrorData
from tokenflood.models.data.llm_request_data import LLMRequestData
from tokenflood.models.message_list import MessageList, chat_schema
from tokenflood.models.data.ping_request_data import PingData
from tokenflood.models.util import get_fields


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


def get_relative_file_path(reference_file: str, target_file_name: str):
    """Return the path to another file from a python file's __file__ path."""
    file_path = os.path.abspath(reference_file)
    module_dir = os.path.dirname(file_path)
    return os.path.join(module_dir, target_file_name)


def write_file(filename: str, s: str):
    with open(filename, "w") as f:
        f.write(s)


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def read_jsonl_messages(file_name: str) -> list[MessageList]:
    with open(file_name) as f:
        data = f.read()

    lines = data.splitlines()
    message_lists = []
    for idx, line in enumerate(lines, start=1):
        try:
            message_list = json.loads(line)["messages"]
            validate(instance=message_list, schema=chat_schema)
            message_lists.append(message_list)
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
            raise ValueError(f"{file_name}:{idx}:{str(e)}")

    return message_lists


def folder_contains_file(folder: str, filename: str) -> bool:
    target_file = os.path.join(folder, filename)
    return os.path.isdir(folder) and os.path.isfile(target_file)


def folder_contains_files(folder: str, filenames: Set[str]) -> bool:
    for f in filenames:
        if not folder_contains_file(folder, f):
            return False
    return True


def is_observation_result_folder(folder: str) -> bool:
    return folder_contains_files(folder, COMMON_RESULT_FILES) and folder_contains_files(
        folder, OBSERVATION_RESULT_FILES
    )


def is_load_test_result_folder(folder) -> bool:
    return folder_contains_files(folder, COMMON_RESULT_FILES) and folder_contains_files(
        folder, LOAD_TEST_RESULT_FILES
    )


class FileSink:
    def __init__(self, destination: str):
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.destination = destination
        self.consumer_task = None
        self.closed = False

    async def _consume(self):
        async with aiofiles.open(self.destination, "w", encoding="utf-8") as f:
            while True:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=2)
                    if isinstance(item, str):
                        await f.write(item)
                        await f.flush()
                        self.queue.task_done()
                    else:
                        self.queue.task_done()
                        break
                except asyncio.TimeoutError:
                    pass

    def write(self, item: str):
        if self.closed:
            raise RuntimeError(
                f"Cannot write to FileSink for {self.destination} that was already closed"
            )
        self.queue.put_nowait(item)

    def close(self):
        self.closed = True
        self.queue.put_nowait(None)

    def activate(self):
        self.consumer_task = asyncio.create_task(self._consume())

    async def wait_for_pending_writes(self):
        await self.queue.join()
        # await self.consumer_task


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

    async def wait_for_pending_writes(self):
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
        await asyncio.sleep(0.1)
        await self.error_sink.wait_for_pending_writes()
        await self.llm_request_sink.wait_for_pending_writes()
        await self.network_latency_sink.wait_for_pending_writes()

    def close(self):
        self.error_sink.close()
        self.llm_request_sink.close()
        self.network_latency_sink.close()


def read_prompts(
    files: list[str], file_format: Literal["chat", "text"]
) -> list[MessageList]:
    message_lists = []
    for prompt_file in files:
        if file_format == "text":
            prompt = read_file(prompt_file)
            message_lists.extend([create_message_list_from_prompt(prompt)])
        elif file_format == "chat":
            message_lists.extend(read_jsonl_messages(prompt_file))
        else:
            raise ValueError(f"Bad prompt file format: {file_format}")
    return message_lists
