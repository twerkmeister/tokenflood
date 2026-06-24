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
    LOAD_TEST_SPEC_FILE,
)
from tokenflood.io import (
    FileIOContext,
    read_endpoint_spec,
    read_observation_spec,
    read_load_test_spec,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.models.run_specs.load_test_spec import LoadTestSpec
from tokenflood.networking import (
    ObserveURLMiddleware,
    patch_aiohttp_client_session,
    unpatch_aiohttp_client_session,
)


def join_folder_checked(path: str, folder: str) -> str:
    new_folder = os.path.join(path, folder)
    assert os.path.isdir(new_folder)
    return new_folder


@pytest.fixture(scope="session")
def test_folder() -> str:
    folder = os.path.dirname(__file__)
    assert folder.endswith(os.path.join("tokenflood", "tests"))
    assert os.path.isdir(folder)
    return folder


@pytest.fixture(scope="session")
def data_folder(test_folder: str) -> str:
    return join_folder_checked(test_folder, "data")


@pytest.fixture(scope="session")
def run_suites_folder(data_folder: str) -> str:
    return join_folder_checked(data_folder, "load_test_specs")


@pytest.fixture(scope="session")
def observation_specs_folder(data_folder: str) -> str:
    return join_folder_checked(data_folder, "observation_specs")


@pytest.fixture(scope="session")
def endpoint_specs_folder(data_folder: str) -> str:
    return join_folder_checked(data_folder, "endpoint_specs")


@pytest.fixture(scope="session")
def results_folder(data_folder: str) -> str:
    return join_folder_checked(data_folder, "testing_results")

@pytest.fixture(scope="session")
def diverse_results_folder(data_folder: str) -> str:
    return join_folder_checked(data_folder, "diverse_result_folders")


@pytest.fixture(scope="session")
def prompts_folder(data_folder: str) -> str:
    return join_folder_checked(data_folder, "prompts")


@pytest.fixture(scope="session")
def load_test_results_folder(results_folder: str) -> str:
    return join_folder_checked(results_folder, "load_test_results")


@pytest.fixture(scope="session")
def observation_results_folder(results_folder: str) -> str:
    return join_folder_checked(results_folder, "observation_results")


@pytest.fixture
def short_observation_spec(observation_specs_folder) -> ObservationSpec:
    filename = os.path.join(observation_specs_folder, "short.yml")
    return read_observation_spec(filename)


@pytest.fixture
def superfast_observation_spec(observation_specs_folder) -> ObservationSpec:
    filename = os.path.join(observation_specs_folder, "superfast.yml")
    return read_observation_spec(filename)


@pytest.fixture
def base_load_test_spec(run_suites_folder) -> LoadTestSpec:
    filename = os.path.join(run_suites_folder, "base.yml")
    return read_load_test_spec(filename)


@pytest.fixture
def tiny_load_test_spec(run_suites_folder) -> LoadTestSpec:
    filename = os.path.join(run_suites_folder, "tiny.yml")
    return read_load_test_spec(filename)


@pytest.fixture
def llm_requests_csv_file(load_test_results_folder) -> str:
    filename = os.path.join(load_test_results_folder, LLM_REQUESTS_FILE)
    assert os.path.isfile(filename)
    return filename


@pytest.fixture
def network_latency_csv_file(load_test_results_folder) -> str:
    filename = os.path.join(load_test_results_folder, NETWORK_LATENCY_FILE)
    assert os.path.isfile(filename)
    return filename


@pytest.fixture
def llm_requests_df(llm_requests_csv_file) -> pd.DataFrame:
    return pd.read_csv(llm_requests_csv_file)


@pytest.fixture
def network_latency_df(network_latency_csv_file) -> pd.DataFrame:
    return pd.read_csv(network_latency_csv_file)


@pytest.fixture
def base_endpoint_spec(endpoint_specs_folder) -> EndpointSpec:
    filename = os.path.join(endpoint_specs_folder, "base.yml")
    return read_endpoint_spec(filename)


@pytest.fixture
def results_endpoint_spec(load_test_results_folder) -> EndpointSpec:
    filename = os.path.join(load_test_results_folder, ENDPOINT_SPEC_FILE)
    return read_endpoint_spec(filename)


@pytest.fixture
def results_run_suite(load_test_results_folder) -> LoadTestSpec:
    filename = os.path.join(load_test_results_folder, LOAD_TEST_SPEC_FILE)
    return read_load_test_spec(filename)


@pytest.fixture
def openai_endpoint_spec() -> EndpointSpec:
    return EndpointSpec(provider="openai", model="gpt-4o-mini")


@pytest.fixture
def tokenizer(base_endpoint_spec) -> Tokenizer:
    return Tokenizer.from_pretrained(base_endpoint_spec.model)


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


@pytest.fixture()
def with_patched_aiohttp_session() -> Generator[None, None, None]:
    patch_aiohttp_client_session()
    yield
    unpatch_aiohttp_client_session()


@pytest.fixture()
def url_observer() -> ObserveURLMiddleware:
    ObserveURLMiddleware.reset()
    return ObserveURLMiddleware()
