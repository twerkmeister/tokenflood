import os

import pandas as pd
from pandas.testing import assert_frame_equal
from tokenflood.constants import REQUESTS_PER_SECOND_COLUMN_NAME
from tokenflood.graphing import (
    visualize_percentiles_across_request_rates,
    write_out_raw_data_points,
)


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


def test_visualize_percentiles_across_request_rates(
    tiny_run_suite, tiny_run_data, unique_temporary_file, tiny_suite_plot_file
):
    visualize_percentiles_across_request_rates(
        tiny_run_suite, tiny_run_data, unique_temporary_file
    )
    assert os.path.exists(tiny_suite_plot_file)
