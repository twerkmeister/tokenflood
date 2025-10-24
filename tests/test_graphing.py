import os

import pandas as pd
from pandas.testing import assert_frame_equal
from tokenflood.constants import (
    ERROR_FILE,
    LATENCY_GRAPH_FILE,
    REQUESTS_PER_SECOND_COLUMN_NAME,
)
from tokenflood.graphing import (
    visualize_percentiles_across_request_rates,
    write_out_error,
    write_out_raw_data_points,
)
from tokenflood.io import read_file


def test_write_out_raw_data_points(tiny_run_data, unique_temporary_file):
    write_out_raw_data_points(tiny_run_data, unique_temporary_file)
    df = pd.read_csv(unique_temporary_file)
    total_len = sum(
        [run_data.run_spec.total_num_requests for run_data in tiny_run_data]
    )
    assert len(df) == total_len
    for i in range(len(tiny_run_data)):
        filtered_df = df[
            df[REQUESTS_PER_SECOND_COLUMN_NAME]
            == tiny_run_data[i].run_spec.requests_per_second
        ]
        comparable_df = filtered_df.drop(
            REQUESTS_PER_SECOND_COLUMN_NAME, axis=1
        ).reset_index(drop=True)
        assert_frame_equal(comparable_df, tiny_run_data[i].results.as_dataframe())


def test_visualize_percentiles_across_request_rates_changes(
    tiny_run_suite, tiny_run_data, tiny_suite_plot_file
):
    visualize_percentiles_across_request_rates(
        tiny_run_suite, tiny_run_data, tiny_suite_plot_file
    )
    assert os.path.exists(tiny_suite_plot_file)


def test_visualize_percentiles_across_request_rates_unfinished_run(
    tiny_run_suite, tiny_run_data, unique_temporary_folder, monkeypatch
):
    monkeypatch.chdir(unique_temporary_folder)
    file = LATENCY_GRAPH_FILE
    tiny_run_data = tiny_run_data[:1]
    assert len(tiny_run_data) < len(tiny_run_suite.requests_per_second_rates)
    visualize_percentiles_across_request_rates(tiny_run_suite, tiny_run_data, file)
    assert os.path.exists(file)


def test_write_out_error(unique_temporary_folder, tiny_run_data):
    target_file = os.path.join(unique_temporary_folder, ERROR_FILE)
    error_str = "test error"
    tiny_run_data_with_error = [
        tiny_run_data[0],
        tiny_run_data[1].model_copy(update={"error": error_str}),
    ]
    write_out_error(tiny_run_data_with_error, target_file)
    assert error_str == read_file(target_file)


def test_write_out_no_error(unique_temporary_folder, tiny_run_data):
    target_file = os.path.join(unique_temporary_folder, ERROR_FILE)
    write_out_error(tiny_run_data, target_file)
    assert not os.path.exists(target_file)
