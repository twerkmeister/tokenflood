import os
import pytest

@pytest.fixture(scope="session")
def test_folder() -> str:
    folder = os.path.dirname(__file__)
    assert folder.endswith(os.path.join("tokenflood", "tests"))
    assert os.path.exists(folder)
    return folder
