from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.results import Results


@pytest.fixture()
def results_default_kwargs() -> Dict:
    default_results = Results(
        prompts=("A", "A"),
        generated_texts=("B", "C"),
        latencies=(100, 120),
        expected_input_lengths=(1024, 1024),
        expected_prefix_lengths=(128, 140),
        expected_output_lengths=(16, 12),
        measured_input_lengths=(512, 514),
        measured_prefix_lengths=(100, 112),
        measured_output_lengths=(16, 12),
    )
    return default_results.model_dump()


@pytest.mark.parametrize(
    "kwarg_changes, expectation",
    [
        (
            {},
            does_not_raise(),
        ),
        (
            {"latencies": (100,)},
            pytest.raises(ValueError),
        ),
        (
            {"expected_input_lengths": ()},
            pytest.raises(ValueError),
        ),
        (
            {"latencies": (-100, 120)},
            pytest.raises(ValueError),
        ),
        (
            {"expected_input_lengths": (-1024, 1024)},
            pytest.raises(ValueError),
        ),
        (
            {"expected_prefix_lengths": (-128, 140)},
            pytest.raises(ValueError),
        ),
        (
            {"expected_output_lengths": (-16, 12)},
            pytest.raises(ValueError),
        ),
        (
            {"measured_input_lengths": (-512, 514)},
            pytest.raises(ValueError),
        ),
        (
            {"measured_prefix_lengths": (-100, 112)},
            pytest.raises(ValueError),
        ),
        (
            {"measured_output_lengths": (-16, 12)},
            pytest.raises(ValueError),
        ),
    ],
)
def test_result_validation(kwarg_changes, expectation, results_default_kwargs):
    with expectation:
        Results(**{**results_default_kwargs, **kwarg_changes})


def test_get_input_length_error(results_default_kwargs):
    results = Results(**results_default_kwargs)
    assert results.get_input_length_error() == 511.0


def test_get_prefix_length_error(results_default_kwargs):
    results = Results(**results_default_kwargs)
    assert results.get_prefix_length_error() == 28.0


def test_get_output_length_error(results_default_kwargs):
    results = Results(**results_default_kwargs)
    assert results.get_output_length_error() == 0.0


def test_get_latency_percentile(results_default_kwargs):
    results = Results(**results_default_kwargs)
    assert results.get_latency_percentile(50) == 110.0
