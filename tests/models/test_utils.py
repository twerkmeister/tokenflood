from tokenflood.models.load_type import LoadType
from tokenflood.models.util import get_fields


def test_get_fields():
    assert get_fields(LoadType) == [
        "prompt_length",
        "prefix_length",
        "output_length",
        "weight",
    ]
