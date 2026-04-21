import pytest
from tokenflood.models.load_types.token_set import TokenSet
from tests.utils import does_not_raise


@pytest.mark.parametrize(
    "tokens, expectation",
    [
        (("A", "B"), does_not_raise()),
        (("A", "A"), pytest.raises(ValueError)),
        (tuple(), pytest.raises(ValueError)),
        (("A", ""), pytest.raises(ValueError)),
        (("A",), pytest.raises(ValueError)),
    ],
)
def test_token_set_validation(tokens, expectation):
    with expectation:
        TokenSet(tokens=tokens)
