import numpy as np
import pytest

from tokenflood.analysis import (
    aggregate,
    calculate_percentile,
    extend_group_stats,
    get_group_data,
    get_group_stats,
    get_groups,
    get_percentile_float,
    mean_float,
    mean_int,
)


def test_get_groups(llm_requests_df):
    assert get_groups(llm_requests_df) == ["0", "1"]


def test_get_group_data(llm_requests_df):
    assert 2 == len(get_group_data(llm_requests_df, "0"))
    assert 4 == len(get_group_data(llm_requests_df, "1"))


def test_aggregate(llm_requests_df):
    field = "latency"
    assert round(np.average(llm_requests_df[field]), 2) == aggregate(
        llm_requests_df, field, mean_float
    )
    assert int(np.average(llm_requests_df[field])) == aggregate(
        llm_requests_df, field, mean_int
    )


def test_get_group_stats(llm_requests_df):
    field = "request_number"
    funcs = [mean_int, mean_float, calculate_percentile(0), calculate_percentile(100)]
    group_stats = get_group_stats(llm_requests_df, field, funcs)
    assert len(group_stats) == 2
    assert group_stats["0"] == [0, 0.5, 0, 1]
    assert group_stats["1"] == [1, 1.5, 0, 3]


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


@pytest.mark.parametrize(
    "stats1, stats2, result_stats",
    [
        ({"a": [1, 2]}, {"a": [3]}, {"a": [1, 2, 3]}),
        ({"a": [1], "b": [1]}, {"a": [2], "b": [3]}, {"a": [1, 2], "b": [1, 3]}),
    ],
)
def test_extend_group_stats(stats1, stats2, result_stats):
    assert extend_group_stats(stats1, stats2) == result_stats
