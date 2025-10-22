import os
import pytest
from tokenizers import Tokenizer

from tokenflood.io import read_endpoint_spec, read_run_suite
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import LoadType
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


@pytest.fixture(scope="session")
def model_id(base_endpoint_spec) -> str:
    return base_endpoint_spec.model


@pytest.fixture(scope="session")
def tokenizer(model_id: str) -> Tokenizer:
    return Tokenizer.from_pretrained(model_id)


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
