import os
import tempfile

from tokenflood.io import (
    make_run_folder,
    read_pydantic_yaml_list,
    read_run_suite,
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


def test_make_run_folder():
    tmpdir = tempfile.mkdtemp()
    result_folder = os.path.join(tmpdir, "results")
    assert not os.path.exists(result_folder)
    make_run_folder(result_folder)
    assert os.path.isdir(result_folder)
