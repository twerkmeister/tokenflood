import pytest

from tokenflood.schedule import burstiness_to_burstiness_control


@pytest.mark.parametrize(
    "burstiness, expected_control",
    [
        (10, 1),
        (9, 3),
        (8, 5),
        (7, 7),
        (6, 9),
        (5, 11),
        (4, 13),
        (3, 15),
        (2, 17),
        (1, 19),
        (0, 21),
    ],
)
def test_burstiness_to_burstiness_control(burstiness, expected_control):
    assert burstiness_to_burstiness_control(burstiness) == expected_control
