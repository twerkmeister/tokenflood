import os
import pytest

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
