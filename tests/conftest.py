import os
import pytest
from tokenizers.tokenizers import Tokenizer

from tokenflood.models.endpoint_spec import EndpointSpec


@pytest.fixture(scope="session")
def test_folder() -> str:
    folder = os.path.dirname(__file__)
    assert folder.endswith(os.path.join("tokenflood", "tests"))
    assert os.path.exists(folder)
    return folder

@pytest.fixture(scope="session")
def test_data_folder(test_folder: str) -> str:
    folder = os.path.join(test_folder, "data")
    assert os.path.exists(folder)
    return folder

@pytest.fixture(scope="session")
def test_specs_folder(test_data_folder: str) -> str:
    folder = os.path.join(test_data_folder, "test_specs")
    assert os.path.exists(folder)
    return folder

@pytest.fixture(scope="session")
def endpoint_specs_folder(test_data_folder: str) -> str:
    folder = os.path.join(test_data_folder, "endpoint_specs")
    assert os.path.exists(folder)
    return folder

@pytest.fixture(scope="session")
def test_model_id() -> str:
    return "HuggingFaceTB/SmolLM-135M-Instruct"

@pytest.fixture(scope="session")
def test_tokenizer(test_model_id: str) -> Tokenizer:
    return Tokenizer.from_pretrained(test_model_id)

@pytest.fixture(scope="session")
def test_endpoint_spec(test_model_id: str) -> EndpointSpec:
    return EndpointSpec(model=f"openai/{test_model_id}", base_url="http://127.0.0.1:8000/v1")
