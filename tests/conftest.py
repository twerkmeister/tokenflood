import os
import shutil
import tempfile
from typing import Generator, List

import pytest
from tokenizers import Tokenizer

from tokenflood.io import read_endpoint_spec, read_pydantic_yaml_list, read_run_suite
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import LoadType
from tokenflood.models.run_data import RunData
from tokenflood.models.run_spec import HeuristicRunSpec
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.token_set import TokenSet


@pytest.fixture(scope="session")
def test_folder() -> str:
    folder = os.path.dirname(__file__)
    assert folder.endswith(os.path.join("tokenflood", "tests"))
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def data_folder(test_folder: str) -> str:
    folder = os.path.join(test_folder, "data")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def run_suites_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "run_suites")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def endpoint_specs_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "endpoint_specs")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def run_data_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "run_data")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def plots_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "plots")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def tiny_suite_plot_file(plots_folder: str) -> str:
    file = os.path.join(plots_folder, "tiny_suite_result.png")
    return file


@pytest.fixture(scope="session")
def tiny_run_data_file_unsafe(run_data_folder: str) -> str:
    file = os.path.join(run_data_folder, "tiny_run_data.yml")
    return file


@pytest.fixture(scope="session")
def tiny_run_data_file(tiny_run_data_file_unsafe: str) -> str:
    assert os.path.isfile(tiny_run_data_file_unsafe)
    return tiny_run_data_file_unsafe


@pytest.fixture(scope="session")
def tiny_run_data(tiny_run_data_file) -> List[RunData]:
    return read_pydantic_yaml_list(RunData)(tiny_run_data_file)


@pytest.fixture
def base_run_suite(run_suites_folder) -> HeuristicRunSuite:
    filename = os.path.join(run_suites_folder, "base.yml")
    return read_run_suite(filename)


@pytest.fixture
def tiny_run_suite(run_suites_folder) -> HeuristicRunSuite:
    filename = os.path.join(run_suites_folder, "tiny.yml")
    return read_run_suite(filename)


@pytest.fixture
def base_endpoint_spec(endpoint_specs_folder) -> EndpointSpec:
    filename = os.path.join(endpoint_specs_folder, "base.yml")
    return read_endpoint_spec(filename)


@pytest.fixture
def openai_endpoint_spec() -> EndpointSpec:
    return EndpointSpec(provider="openai", model="gpt-4o-mini")


@pytest.fixture
def tokenizer(base_endpoint_spec) -> Tokenizer:
    return Tokenizer.from_pretrained(base_endpoint_spec.model)


@pytest.fixture(scope="session")
def run_spec() -> HeuristicRunSpec:
    return HeuristicRunSpec(
        name="abc",
        requests_per_second=1,
        test_length_in_seconds=2,
        load_types=(LoadType(prompt_length=128, prefix_length=32, output_length=2),),
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
