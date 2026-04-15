import numpy as np
import pytest

from tokenflood.analysis import (
    aggregate,
    get_group_data,
    get_group_ids,
    get_percentile_float,
    Mean,
)


def test_get_groups(llm_requests_df):
    assert get_group_ids(llm_requests_df) == ["0", "1"]


def test_get_group_data(llm_requests_df):
    assert 2 == len(get_group_data(llm_requests_df, "0"))
    assert 4 == len(get_group_data(llm_requests_df, "1"))


def test_aggregate(llm_requests_df):
    field = "latency"
    assert round(np.average(llm_requests_df[field]), 2) == aggregate(
        llm_requests_df, field, Mean
    )


@pytest.mark.parametrize(
    "sequence, percentile, expected_result",
    [
        ([], 25, 0.0),
        ([], 75, 0.0),
        ([1], 20, 1.0),
        ([1], 80, 1.0),
        ([0, 2, 4, 6, 8, 10], 50, 5.0),
    ],
)
def test_get_percentile_float(sequence, expected_result, percentile):
    assert expected_result == get_percentile_float(sequence, percentile)
