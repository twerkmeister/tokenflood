import os
import shutil
import sys

import requests

from tokenflood.cli import (
    create_starter_files,
    main,
    parse_args,
    run,
    start_visualization,
)
from tokenflood.constants import (
    ENDPOINT_SPEC_FILE,
    OBSERVATION_SPEC_FILE,
    RESULTS_FOLDER,
    LOAD_SPEC_FILE,
)
from tokenflood.io import (
    is_observation_result_folder,
    is_load_result_folder,
    list_dir_relative,
    read_endpoint_spec,
    read_observation_spec,
    read_load_spec,
    write_pydantic_yaml,
)
from tokenflood.starter_pack import (
    starter_endpoint_spec_vllm,
    starter_observation_spec,
    starter_run_suite,
)


def test_parse_args_run():
    endpoint_spec = "endpoint.yml"
    run_spec = "load.yml"
    args = parse_args(["run", run_spec, endpoint_spec])
    assert args.endpoint == endpoint_spec
    assert args.run_spec == run_spec
    assert args.func.__name__ == run.__name__


def test_parse_args_init(unique_temporary_folder, monkeypatch):
    args = parse_args(["init"])
    assert args.func.__name__ == create_starter_files.__name__


def test_parse_args_empty():
    args = parse_args([])
    assert args.func.__name__ == "print_help"


def test_init(unique_temporary_folder, monkeypatch):
    monkeypatch.chdir(unique_temporary_folder)
    args = parse_args(["init"])
    create_starter_files(args)
    os.path.isfile(LOAD_SPEC_FILE)
    os.path.isfile(ENDPOINT_SPEC_FILE)
    os.path.isfile(OBSERVATION_SPEC_FILE)
    run_suite = read_load_spec(LOAD_SPEC_FILE)
    endpoint_spec = read_endpoint_spec(ENDPOINT_SPEC_FILE)
    observation_spec = read_observation_spec(OBSERVATION_SPEC_FILE)
    assert run_suite == starter_run_suite
    assert endpoint_spec == starter_endpoint_spec_vllm
    assert observation_spec == starter_observation_spec


def test_init_does_not_override(unique_temporary_folder, monkeypatch):
    monkeypatch.chdir(unique_temporary_folder)
    args = parse_args(["init"])
    create_starter_files(args)
    assert len(os.listdir(unique_temporary_folder)) == 3
    create_starter_files(args)
    assert len(os.listdir(unique_temporary_folder)) == 6


def test_load_test(
    monkeypatch, unique_temporary_folder, tiny_load_spec, base_endpoint_spec
):
    monkeypatch.chdir(unique_temporary_folder)
    write_pydantic_yaml(LOAD_SPEC_FILE, tiny_load_spec)
    write_pydantic_yaml(ENDPOINT_SPEC_FILE, base_endpoint_spec)
    args = parse_args(["run", LOAD_SPEC_FILE, ENDPOINT_SPEC_FILE, "-y"])
    run(args)
    assert os.path.exists(RESULTS_FOLDER)
    run_folders = list_dir_relative(RESULTS_FOLDER)
    assert len(run_folders) == 1
    assert is_load_result_folder(run_folders[0])


def test_load_test_decline(
    monkeypatch, unique_temporary_folder, tiny_load_spec, base_endpoint_spec
):
    monkeypatch.chdir(unique_temporary_folder)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    write_pydantic_yaml(LOAD_SPEC_FILE, tiny_load_spec)
    write_pydantic_yaml(ENDPOINT_SPEC_FILE, base_endpoint_spec)
    args = parse_args(["run", LOAD_SPEC_FILE, ENDPOINT_SPEC_FILE])
    run(args)
    assert not os.path.exists(RESULTS_FOLDER)


def test_observe_endpoint(
    monkeypatch, unique_temporary_folder, superfast_observation_spec, base_endpoint_spec
):
    monkeypatch.chdir(unique_temporary_folder)

    write_pydantic_yaml(OBSERVATION_SPEC_FILE, superfast_observation_spec)
    write_pydantic_yaml(ENDPOINT_SPEC_FILE, base_endpoint_spec)
    args = parse_args(["run", OBSERVATION_SPEC_FILE, ENDPOINT_SPEC_FILE, "-y"])
    run(args)
    assert os.path.exists(RESULTS_FOLDER)
    run_folders = list_dir_relative(RESULTS_FOLDER)
    assert len(run_folders) == 1
    assert is_observation_result_folder(run_folders[0])


def test_observe_endpoint_decline(
    monkeypatch, unique_temporary_folder, superfast_observation_spec, base_endpoint_spec
):
    monkeypatch.chdir(unique_temporary_folder)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    write_pydantic_yaml(OBSERVATION_SPEC_FILE, superfast_observation_spec)
    write_pydantic_yaml(ENDPOINT_SPEC_FILE, base_endpoint_spec)
    args = parse_args(["run", OBSERVATION_SPEC_FILE, ENDPOINT_SPEC_FILE])
    run(args)
    assert not os.path.exists(RESULTS_FOLDER)


def test_load_dotenv(unique_temporary_folder, monkeypatch):
    monkeypatch.chdir(unique_temporary_folder)
    env_var = "TEST_X_ABC"
    env_value = "123"
    with open(".env", "w") as f:
        f.write(f"{env_var}={env_value}\n")

    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [sys.argv[0]])
        main()
        assert os.getenv(env_var) == env_value


def test_start_visualization(monkeypatch, unique_temporary_folder, results_folder):
    monkeypatch.chdir(unique_temporary_folder)
    copy_of_results_folder = os.path.join(unique_temporary_folder, "results")
    shutil.copytree(results_folder, copy_of_results_folder)
    assert "results" in os.listdir(unique_temporary_folder)
    assert "load_results" in os.listdir(copy_of_results_folder)

    args = parse_args(["viz"])
    app, url = start_visualization(args, False, False)

    response = requests.get(url)
    assert response.status_code == 200
