import os
import pytest
from tokenizers import Tokenizer

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.run_spec import HeuristicRunSpec
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
def run_specs_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "run_specs")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def endpoint_specs_folder(data_folder: str) -> str:
    folder = os.path.join(data_folder, "endpoint_specs")
    assert os.path.exists(folder)
    return folder


@pytest.fixture(scope="session")
def model_id() -> str:
    return "HuggingFaceTB/SmolLM-135M-Instruct"


@pytest.fixture(scope="session")
def tokenizer(model_id: str) -> Tokenizer:
    return Tokenizer.from_pretrained(model_id)


@pytest.fixture(scope="session")
def endpoint_spec(model_id: str) -> EndpointSpec:
    return EndpointSpec(model=f"openai/{model_id}", base_url="http://127.0.0.1:8000/v1")


@pytest.fixture(scope="session")
def run_spec() -> HeuristicRunSpec:
    return HeuristicRunSpec(
        name="abc",
        requests_per_second=1,
        test_length_in_seconds=2,
        prompt_lengths=(64,),
        output_lengths=(2,),
        prefix_lengths=(32,),
    )


@pytest.fixture(scope="session")
def token_set() -> TokenSet:
    return TokenSet(tokens=(" A", " B", " C", " D", " E"))


@pytest.fixture(scope="session")
def heuristic_task() -> HeuristicTask:
    return HeuristicTask(task="Ignore the random input and write a letter to Santa.")
