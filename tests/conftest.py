import os
import shutil
import tempfile
from typing import Generator

import pandas as pd
import pytest
from tokenizers import Tokenizer

from tokenflood.constants import (
    ENDPOINT_SPEC_FILE,
    ERROR_FILE,
    LLM_REQUESTS_FILE,
    NETWORK_LATENCY_FILE,
    RUN_SUITE_FILE,
    SUMMARY_FILE,
)
from tokenflood.io import (
    FileIOContext,
    read_endpoint_spec,
    read_pydantic_yaml,
    read_run_suite,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import LoadType
from tokenflood.models.run_spec import HeuristicRunSpec
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.run_summary import RunSummary
from tokenflood.models.token_set import TokenSet


@pytest.fixture(scope="session")
def test_folder() -> str:
    folder = os.path.dirname(__file__)
    assert folder.endswith(os.path.join("tokenflood", "tests"))
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def data_folder(test_folder: str) -> str:
    folder = os.path.join(test_folder, "data")
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def run_suites_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "run_suites")
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def endpoint_specs_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "endpoint_specs")
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def plots_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "plots")
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def results_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "test_results")
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def results_plot_file(results_folder: str) -> str:
    file = os.path.join(results_folder, "tiny_suite_result.png")
    return file


@pytest.fixture
def base_run_suite(run_suites_folder) -> HeuristicRunSuite:
    filename = os.path.join(run_suites_folder, "base.yml")
    return read_run_suite(filename)


@pytest.fixture
def tiny_run_suite(run_suites_folder) -> HeuristicRunSuite:
    filename = os.path.join(run_suites_folder, "tiny.yml")
    return read_run_suite(filename)


@pytest.fixture
def llm_requests_csv_file(results_folder) -> str:
    filename = os.path.join(results_folder, LLM_REQUESTS_FILE)
    assert os.path.isfile(filename)
    return filename


@pytest.fixture
def network_latency_csv_file(results_folder) -> str:
    filename = os.path.join(results_folder, NETWORK_LATENCY_FILE)
    assert os.path.isfile(filename)
    return filename


@pytest.fixture
def summary_file(results_folder) -> str:
    filename = os.path.join(results_folder, SUMMARY_FILE)
    assert os.path.isfile(filename)
    return filename


@pytest.fixture
def llm_requests_df(llm_requests_csv_file) -> pd.DataFrame:
    return pd.read_csv(llm_requests_csv_file)


@pytest.fixture
def network_latency_df(network_latency_csv_file) -> pd.DataFrame:
    return pd.read_csv(network_latency_csv_file)


@pytest.fixture
def results_run_summary(summary_file) -> RunSummary:
    return read_pydantic_yaml(RunSummary)(summary_file)


@pytest.fixture
def base_endpoint_spec(endpoint_specs_folder) -> EndpointSpec:
    filename = os.path.join(endpoint_specs_folder, "base.yml")
    return read_endpoint_spec(filename)


@pytest.fixture
def results_endpoint_spec(results_folder) -> EndpointSpec:
    filename = os.path.join(results_folder, ENDPOINT_SPEC_FILE)
    return read_endpoint_spec(filename)


@pytest.fixture
def results_run_suite(results_folder) -> HeuristicRunSuite:
    filename = os.path.join(results_folder, RUN_SUITE_FILE)
    return read_run_suite(filename)


@pytest.fixture
def openai_endpoint_spec() -> EndpointSpec:
    return EndpointSpec(provider="openai", model="gpt-4o-mini")


@pytest.fixture
def tokenizer(base_endpoint_spec) -> Tokenizer:
    return Tokenizer.from_pretrained(base_endpoint_spec.model)


@pytest.fixture(scope="session")
def run_spec() -> HeuristicRunSpec:
    return HeuristicRunSpec(
        requests_per_second=2,
        test_length_in_seconds=1,
        load_types=(LoadType(prompt_length=128, prefix_length=32, output_length=1),),
    )


@pytest.fixture
def file_io_context(unique_temporary_folder) -> FileIOContext:
    error_file = os.path.join(unique_temporary_folder, ERROR_FILE)
    llm_request_file = os.path.join(unique_temporary_folder, LLM_REQUESTS_FILE)
    network_latency_file = os.path.join(unique_temporary_folder, NETWORK_LATENCY_FILE)
    return FileIOContext(
        llm_request_file=llm_request_file,
        network_latency_file=network_latency_file,
        error_file=error_file,
    )


@pytest.fixture(scope="session")
def token_set() -> TokenSet:
    return TokenSet(tokens=(" A", " B", " C", " D", " E"))


@pytest.fixture(scope="session")
def heuristic_task() -> HeuristicTask:
    return HeuristicTask(task="Ignore the random input and write a letter to Santa.")


@pytest.fixture()
def unique_temporary_file() -> Generator[str, None, None]:
    _, f_name = tempfile.mkstemp()
    assert os.path.isfile(f_name)
    yield f_name
    os.remove(f_name)
    assert not os.path.isfile(f_name)


@pytest.fixture()
def unique_temporary_folder() -> Generator[str, None, None]:
    name = tempfile.mkdtemp()
    assert os.path.isdir(name)
    yield name
    shutil.rmtree(name)
    assert not os.path.isdir(name)
