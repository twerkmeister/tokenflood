import numpy as np
import pytest

from tests.utils import does_not_raise
from tokenflood.models.test_spec import HeuristicTestSpec, TestSpec


@pytest.mark.parametrize(
    "name, requests_per_second, test_length_in_seconds, expectation",
    [
        ("abc", 3, 10, does_not_raise()),
        ("", 3, 10, pytest.raises(ValueError)),
        ("abc", 0, 10, pytest.raises(ValueError)),
        ("abc", -1, 10, pytest.raises(ValueError)),
        ("abc", -3, 0, pytest.raises(ValueError)),
        ("abc", -3, -1, pytest.raises(ValueError)),
        ("abc", 0.1, 1, pytest.raises(ValueError)),
    ],
)
def test_test_spec_validation(
    name, requests_per_second, test_length_in_seconds, expectation
):
    with expectation:
        TestSpec(
            name=name,
            requests_per_second=requests_per_second,
            test_length_in_seconds=test_length_in_seconds,
        )


@pytest.mark.parametrize(
    "prompt_lengths, output_lengths, prefix_lengths, expectation",
    [
        ([1000], [20], [0], does_not_raise()),
        ([1000, 1000], [20, 20], [0, 0], does_not_raise()),
        ([1000, 1200], [20], [0], does_not_raise()),
        ([1000], [20], [], pytest.raises(ValueError)),
        ([1000], [], [0], pytest.raises(ValueError)),
        ([], [20], [0], pytest.raises(ValueError)),
        ([-1], [20], [0], pytest.raises(ValueError)),
        ([1000], [-20], [0], pytest.raises(ValueError)),
        ([1000], [20], [-40], pytest.raises(ValueError)),
    ],
)
def test_heuristic_test_spec_validation(
    prompt_lengths, output_lengths, prefix_lengths, expectation
):
    with expectation:
        HeuristicTestSpec(
            name="abc",
            requests_per_second=3,
            test_length_in_seconds=10,
            prompt_lengths=prompt_lengths,
            prefix_lengths=prefix_lengths,
            output_lengths=output_lengths,
        )


def test_heuristic_test_spec_sampling():
    spec = HeuristicTestSpec(
        name="abc",
        requests_per_second=100,
        test_length_in_seconds=1000,
        prompt_lengths=[100, 100, 112],
        prefix_lengths=[20],
        output_lengths=[12, 12, 12, 6],
    )

    prompt_lengths, prefix_lengths = spec.sample_input_tokens()
    output_lengths = spec.sample_output_tokens()
    assert np.allclose(
        np.average(spec.prompt_lengths), np.average(prompt_lengths), atol=1
    )
    assert np.allclose(
        np.average(spec.prefix_lengths), np.average(prefix_lengths), atol=1
    )
    assert np.allclose(
        np.average(spec.output_lengths), np.average(output_lengths), atol=1
    )
