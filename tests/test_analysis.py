import os

import pandas as pd
import pytest

from tokenflood.constants import (
    LATENCY_GRAPH_FILE,
)
from tokenflood.analysis import (
    create_summary,
    get_percentile_float,
    make_super_title,
    visualize_percentiles_across_request_rates,
)
from tokenflood.models.run_summary import RunSummary
from tokenflood.util import get_date_str


def test_visualize_percentiles_across_request_rates_changes(
    results_run_suite, results_endpoint_spec, results_run_summary, results_plot_file
):
    visualize_percentiles_across_request_rates(
        make_super_title(
            results_run_suite,
            results_endpoint_spec,
            get_date_str(),
            results_run_summary,
        ),
        results_run_summary,
        results_plot_file,
    )
    assert os.path.exists(results_plot_file)


def test_visualize_percentiles_across_request_rates_empty_run_summary(
    results_run_suite,
    results_endpoint_spec,
    results_run_summary,
    unique_temporary_folder,
    monkeypatch,
):
    monkeypatch.chdir(unique_temporary_folder)
    file = LATENCY_GRAPH_FILE
    run_summary = RunSummary.create_empty(
        results_run_suite.name, results_endpoint_spec.provider_model_str
    )
    visualize_percentiles_across_request_rates(
        make_super_title(
            results_run_suite,
            results_endpoint_spec,
            get_date_str(),
            results_run_summary,
        ),
        run_summary,
        file,
    )
    assert os.path.exists(file)


def test_create_summary(
    results_run_suite,
    results_endpoint_spec,
    results_run_summary,
    llm_requests_df,
    network_latency_df,
):
    assert results_run_summary == create_summary(
        results_run_suite, results_endpoint_spec, llm_requests_df, network_latency_df
    )


def test_create_empty_summary(results_run_suite, results_endpoint_spec):
    summary = create_summary(
        results_run_suite, results_endpoint_spec, pd.DataFrame(), pd.DataFrame()
    )
    assert summary.total_num_requests == 0
    assert summary.run_suite == results_run_suite.name
    assert summary.endpoint == results_endpoint_spec.provider_model_str


@pytest.mark.parametrize(
    "seq, percentile, expected_result",
    [
        ([], 20, 0.0),
        ([], 50, 0.0),
        ([1], 20, 1.0),
        ([1], 50, 1.0),
        ([1], 80, 1.0),
        ([2], 20, 2.0),
        ([2], 50, 2.0),
        ([2], 80, 2.0),
        ([0, 100], 50, 50.0),
        ([0, 100], 90, 90.0),
    ],
)
def test_get_percentile_float(seq, percentile, expected_result):
    assert expected_result == get_percentile_float(seq, percentile)
