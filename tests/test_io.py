import asyncio
import os

import pytest

from tokenflood.constants import ERROR_RING_BUFFER_SIZE
from tokenflood.io import (
    CSVFileSink,
    FileSink,
    IOContext,
    add_suffix_to_file_name,
    folder_contains_file,
    folder_contains_files,
    get_first_available_filename_like,
    is_observation_result_folder,
    is_run_result_folder,
    list_dir_relative,
    make_run_folder,
    read_file,
    read_pydantic_yaml_list,
    read_run_suite,
    write_file,
    write_pydantic_yaml,
    write_pydantic_yaml_list,
)
from tokenflood.models.run_suite import HeuristicRunSuite


def test_read_base_endpoint_spec(base_endpoint_spec):
    assert base_endpoint_spec.provider == "hosted_vllm"
    assert base_endpoint_spec.model == "HuggingFaceTB/SmolLM-135M-Instruct"
    assert base_endpoint_spec.base_url == "http://127.0.0.1:8000/v1"


def test_read_base_run_suite(base_run_suite):
    assert len(base_run_suite.load_types) == 2
    assert base_run_suite.name == "ABC"
    assert base_run_suite.requests_per_second_rates == tuple(range(1, 5))
    assert base_run_suite.test_length_in_seconds == 30


def test_read_write_pydantic_model(base_run_suite, unique_temporary_file):
    write_pydantic_yaml(unique_temporary_file, base_run_suite)
    re_read_base_run_suite = read_run_suite(unique_temporary_file)
    assert re_read_base_run_suite == base_run_suite


def test_read_write_pydantic_model_list(base_run_suite, unique_temporary_file):
    object_list = [base_run_suite, base_run_suite]
    write_pydantic_yaml_list(unique_temporary_file, object_list)
    re_read_base_run_suites = read_pydantic_yaml_list(HeuristicRunSuite)(
        unique_temporary_file
    )
    assert len(re_read_base_run_suites) == len(object_list)
    assert re_read_base_run_suites[0] == base_run_suite
    assert re_read_base_run_suites[1] == base_run_suite


def test_make_run_folder(unique_temporary_folder):
    result_folder = os.path.join(unique_temporary_folder, "results")
    assert not os.path.exists(result_folder)
    make_run_folder(result_folder)
    assert os.path.isdir(result_folder)


@pytest.mark.parametrize(
    "filename, suffix, expected",
    [
        ("endpoint_spec.yml", "_01", "endpoint_spec_01.yml"),
        (
            "relative_path/endpoint_spec.yml",
            "_02",
            "relative_path/endpoint_spec_02.yml",
        ),
    ],
)
def test_add_suffix_to_file_name(filename, suffix, expected):
    assert add_suffix_to_file_name(filename, suffix) == expected


def test_get_first_available_file(unique_temporary_folder, base_endpoint_spec):
    filename = os.path.join(unique_temporary_folder, "endpoint_spec.yml")
    available_filename = get_first_available_filename_like(filename)
    assert available_filename == filename
    write_pydantic_yaml(available_filename, base_endpoint_spec)
    next_available_filename = get_first_available_filename_like(filename)
    assert next_available_filename == f"{filename[:-4]}_01.yml"
    write_pydantic_yaml(next_available_filename, base_endpoint_spec)
    last_available_filename = get_first_available_filename_like(filename)
    assert last_available_filename == f"{filename[:-4]}_02.yml"


def test_list_dir_relative(unique_temporary_folder, base_endpoint_spec, monkeypatch):
    monkeypatch.chdir(unique_temporary_folder)
    target_dir = "target"
    os.makedirs(target_dir)
    file_name = "endpoint.yml"
    write_pydantic_yaml(os.path.join(target_dir, file_name), base_endpoint_spec)
    files = list_dir_relative(target_dir)
    assert len(files) == 1
    assert files[0] == f"{target_dir}/{file_name}"


def test_write_read_file(unique_temporary_file):
    text = "ABC"
    write_file(unique_temporary_file, text)
    assert text == read_file(unique_temporary_file)


@pytest.mark.asyncio
async def test_file_sink(unique_temporary_file):
    items = ["test\n", "ABC\n"]
    sink = FileSink(unique_temporary_file)
    sink.activate()
    for item in items:
        sink.write(item)
        await asyncio.sleep(0.001)
    sink.close()

    with open(unique_temporary_file) as f:
        assert f.read() == "".join(items)


@pytest.mark.asyncio
async def test_csv_file_sink(unique_temporary_file):
    key1, key2 = "a", "b"
    items = [{key1: 1, key2: 2}, {key2: 4, key1: 3}]
    sink = CSVFileSink(unique_temporary_file, [key1, key2])
    sink.activate()
    for item in items:
        sink.write_dict(item)
        await asyncio.sleep(0.001)
    sink.close()

    with open(unique_temporary_file) as f:
        assert f.read() == "a,b\n1,2\n3,4\n"


@pytest.mark.asyncio
async def test_io_context_abstract_methods():
    io_context = IOContext()

    assert io_context.state_watch.maxlen == ERROR_RING_BUFFER_SIZE
    assert len(io_context.state_watch) == 0

    with pytest.raises(NotImplementedError):
        io_context.write_error({})

    with pytest.raises(NotImplementedError):
        io_context.write_llm_request({})

    with pytest.raises(NotImplementedError):
        io_context.write_network_latency({})

    with pytest.raises(NotImplementedError):
        io_context.activate()

    with pytest.raises(NotImplementedError):
        await io_context.wait_for_pending_writes()

    with pytest.raises(NotImplementedError):
        io_context.close()


def test_read_short_observation_spec(short_observation_spec):
    assert short_observation_spec.duration_hours == 0.03
    assert short_observation_spec.polling_interval_minutes == 1
    assert short_observation_spec.num_requests == 2
    assert short_observation_spec.within_seconds == 1


def test_folder_contains_file(unique_temporary_folder):
    filename = "test.txt"
    assert not folder_contains_file(unique_temporary_folder, filename)
    write_file(os.path.join(unique_temporary_folder, filename), "This is a test")
    assert folder_contains_file(unique_temporary_folder, filename)


def test_folder_contains_files(unique_temporary_folder):
    filenames = ["test.txt", "test2.txt"]
    # before writing any files
    assert not folder_contains_files(unique_temporary_folder, set(filenames))

    # writing one file
    write_file(os.path.join(unique_temporary_folder, filenames[0]), "Test1")
    assert not folder_contains_files(unique_temporary_folder, set(filenames))

    # writing second file
    write_file(os.path.join(unique_temporary_folder, filenames[1]), "Test2")
    assert folder_contains_files(unique_temporary_folder, set(filenames))

    # writing another unrelated file
    write_file(os.path.join(unique_temporary_folder, "test3.xt"), "Test3")
    assert folder_contains_files(unique_temporary_folder, set(filenames))


def test_is_run_result_folder(run_suite_results_folder):
    assert is_run_result_folder(run_suite_results_folder)


def test_is_observation_result_folder(observation_results_folder):
    assert is_observation_result_folder(observation_results_folder)
