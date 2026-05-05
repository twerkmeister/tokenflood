import asyncio
import csv
import os
from collections import deque
from io import StringIO
from typing import Any, Callable, Dict, List, Set, Type, TypeVar, Iterable

import aiofiles
import yaml
from pydantic import BaseModel, TypeAdapter

from tokenflood.constants import (
    COMMON_RESULT_FILES,
    ERROR_RING_BUFFER_SIZE,
    OBSERVATION_RESULT_FILES,
    RESULTS_FOLDER,
    LOAD_TEST_RESULT_FILES,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.data.error_data import ErrorData
from tokenflood.models.data.llm_request_data import LLMRequestData
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.models.data.ping_request_data import PingData
from tokenflood.models.run_specs.load_test_spec import LoadTestSpec
from tokenflood.models.run_specs.typing import SpecificRunSpec
from tokenflood.models.util import get_fields

T = TypeVar("T", bound=BaseModel)


def create_from_basemodel_or_type_adapter(
    data: dict, class_type: Type[T] | TypeAdapter[T]
) -> T:
    if isinstance(class_type, type) and issubclass(class_type, BaseModel):
        return class_type(**data)
    elif isinstance(class_type, TypeAdapter):
        return class_type.validate_python(data)
    else:
        raise ValueError


def read_pydantic_yaml(class_type: Type[T] | TypeAdapter[T]) -> Callable[[str], T]:
    def read_class_type(filename: str) -> T:
        with open(filename) as f:
            data = yaml.safe_load(f)
        return create_from_basemodel_or_type_adapter(data, class_type)

    return read_class_type


def read_pydantic_yaml_list(
    class_type: Type[T] | TypeAdapter[T],
) -> Callable[[str], List[T]]:
    def read_class_type_list(filename: str) -> List[T]:
        with open(filename) as f:
            list_data = yaml.safe_load(f)
        return [
            create_from_basemodel_or_type_adapter(data, class_type)
            for data in list_data
        ]

    return read_class_type_list


class CustomDumper(yaml.SafeDumper):
    def represent_sequence(
        self, tag: str, sequence: Iterable[Any], flow_style: bool | None = None
    ):
        # making sure an iterator argument is not exhausted
        sequence = list(sequence)
        is_simple = all(isinstance(item, (int, float, str, bool)) for item in sequence)
        return super().represent_sequence(tag, sequence, flow_style=is_simple)


def write_pydantic_yaml(filename: str, o: T) -> None:
    with open(filename, "w") as f:
        yaml.dump(
            o.model_dump(),
            f,
            sort_keys=False,
            indent=2,
            Dumper=CustomDumper,
            default_flow_style=False,
        )


def write_pydantic_yaml_list(filename: str, objects: List[T]) -> None:
    with open(filename, "w") as f:
        yaml.dump(
            [o.model_dump() for o in objects],
            f,
            sort_keys=False,
            Dumper=CustomDumper,
            indent=2,
            default_flow_style=False,
        )


def read_endpoint_spec(filename: str) -> EndpointSpec:
    return read_pydantic_yaml(EndpointSpec)(filename)


def read_run_spec(filename: str) -> SpecificRunSpec:
    return read_pydantic_yaml(TypeAdapter(SpecificRunSpec))(filename)


def read_load_test_spec(filename: str) -> LoadTestSpec:
    return read_pydantic_yaml(LoadTestSpec)(filename)


def read_observation_spec(filename: str) -> ObservationSpec:
    return read_pydantic_yaml(ObservationSpec)(filename)


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
            await asyncio.sleep(0.01)


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
        await self.error_sink.wait_for_pending_writes()
        await self.llm_request_sink.wait_for_pending_writes()
        await self.network_latency_sink.wait_for_pending_writes()

    def close(self):
        self.error_sink.close()
        self.llm_request_sink.close()
        self.network_latency_sink.close()
