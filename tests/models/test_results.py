import math
from typing import Dict

import pytest

from tests.utils import does_not_raise
from tokenflood.models.results import Results


@pytest.fixture()
def empty_results() -> Results:
    return Results(
        latencies=(),
        expected_input_lengths=(),
        measured_input_lengths=(),
        expected_prefix_lengths=(),
        measured_prefix_lengths=(),
        expected_output_lengths=(),
        measured_output_lengths=(),
        generated_texts=(),
        prompts=(),
    )


@pytest.fixture()
def default_results() -> Results:
    return Results(
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


@pytest.fixture()
def results_default_kwargs(default_results) -> Dict:
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


def test_get_input_length_error(default_results):
    assert default_results.get_input_length_error() == 511.0


def test_get_input_length_error_empty(empty_results):
    assert math.isnan(empty_results.get_relative_input_length_error())


def test_get_prefix_length_error(default_results):
    assert default_results.get_prefix_length_error() == 28.0


def test_get_prefix_length_error_empty(empty_results):
    assert math.isnan(empty_results.get_relative_prefix_length_error())


def test_get_output_length_error(default_results):
    assert default_results.get_output_length_error() == 0.0


def test_get_output_length_error_empty(empty_results):
    assert math.isnan(empty_results.get_relative_output_length_error())


def test_get_latency_percentile(default_results):
    assert default_results.get_latency_percentile(50) == 110.0


def test_get_latency_percentile_empty(empty_results):
    assert math.isnan(empty_results.get_latency_percentile(50))


def test_as_dataframe(default_results):
    df = default_results.as_dataframe()
    assert len(df) == 2
    assert len(df.columns) == len(default_results.model_dump())
    assert df["latencies"][0] == 100
    assert df.columns[-1] == "prompts"
    assert df.columns[0] == "latencies"
