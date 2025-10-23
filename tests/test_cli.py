import os

from tokenflood.cli import create_starter_files, parse_args, run_and_graph_suite
from tokenflood.constants import LATENCY_GRAPH_FILE, RESULTS_FOLDER, RUN_DATA_FILE
from tokenflood.io import list_dir_relative, read_endpoint_spec, read_run_suite, \
    write_pydantic_yaml
from tokenflood.starter_pack import (
    starter_endpoint_spec_filename,
    starter_endpoint_spec_vllm,
    starter_run_suite,
    starter_run_suite_filename,
)


def test_parse_args_run():
    endpoint_spec = "endpoint.yml"
    run_suite = "suite.yml"
    args = parse_args(["run", run_suite, endpoint_spec])
    assert args.endpoint == endpoint_spec
    assert args.run_suite == run_suite
    assert args.func.__name__ == run_and_graph_suite.__name__


def test_parse_args_ripple(unique_temporary_folder, monkeypatch):
    args = parse_args(["ripple"])
    assert args.func.__name__ == create_starter_files.__name__


def test_parse_args_empty():
    args = parse_args([])
    assert args.func.__name__ == "print_help"


def test_ripple(unique_temporary_folder, monkeypatch):
    monkeypatch.chdir(unique_temporary_folder)
    args = parse_args(["ripple"])
    create_starter_files(args)
    os.path.isfile(starter_run_suite_filename)
    os.path.isfile(starter_endpoint_spec_filename)
    run_suite = read_run_suite(starter_run_suite_filename)
    endpoint_spec = read_endpoint_spec(starter_endpoint_spec_filename)
    assert run_suite == starter_run_suite
    assert endpoint_spec == starter_endpoint_spec_vllm


def test_ripple_does_not_override(unique_temporary_folder, monkeypatch):
    monkeypatch.chdir(unique_temporary_folder)
    args = parse_args(["ripple"])
    create_starter_files(args)
    assert len(os.listdir(unique_temporary_folder)) == 2
    create_starter_files(args)
    assert len(os.listdir(unique_temporary_folder)) == 4


def test_run_and_graph_suite(monkeypatch, unique_temporary_folder, tiny_run_suite, base_endpoint_spec):
    monkeypatch.chdir(unique_temporary_folder)
    write_pydantic_yaml(starter_run_suite_filename, tiny_run_suite)
    write_pydantic_yaml(starter_endpoint_spec_filename, base_endpoint_spec)
    args = parse_args(
        ["run", starter_run_suite_filename, starter_endpoint_spec_filename]
    )
    run_and_graph_suite(args)
    assert os.path.exists(RESULTS_FOLDER)
    run_folders = list_dir_relative(RESULTS_FOLDER)
    assert len(run_folders) == 1
    result_files = os.listdir(run_folders[0])
    assert len(result_files) == 2
    assert set(result_files) == {RUN_DATA_FILE, LATENCY_GRAPH_FILE}
