from tokenflood.models.load_types.load_type import HeuristicLoad
from tokenflood.models.util import get_fields


def test_get_fields():
    assert get_fields(HeuristicLoad) == [
        "type",
        "prompt_length",
        "prefix_length",
        "output_length",
        "task",
        "token_set"
    ]
