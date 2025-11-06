import asyncio
import os

import pytest

from tokenflood.io import (
    CSVFileSink,
    FileSink,
    add_suffix_to_file_name,
    error_to_str,
    get_first_available_filename_like,
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
    assert base_run_suite.percentiles == (50, 90, 98)


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


def test_error_to_str():
    error_text = "Bad value for variable x"
    err = ValueError(error_text)
    assert error_text in error_to_str(err)


def test_error_to_str_none():
    assert error_to_str(None) is None

@pytest.mark.asyncio
async def test_file_sink(unique_temporary_file):
    with open(unique_temporary_file, "w") as f:
        f.write("XYZ\n")

    items = [
        "test\n", "ABC\n"
    ]
    sink = FileSink(unique_temporary_file)
    for item in items:
        sink.write(item)
        await asyncio.sleep(0.001)
    sink.close()

    with open(unique_temporary_file) as f:
        assert f.read() == "XYZ\n" + "".join(items)


@pytest.mark.asyncio
async def test_csv_file_sink(unique_temporary_file):
    key1, key2 = "a", "b"
    items = [
        {key1: 1, key2: 2},
        {key1: 3, key2: 4}
    ]
    sink = CSVFileSink(unique_temporary_file, [key1, key2])
    for item in items:
        sink.write(item)
        await asyncio.sleep(0.001)
    sink.close()

    with open(unique_temporary_file) as f:
        assert f.read() == "a,b\n1,2\n3,4\n"
