import os

from tokenflood.constants import (
    LATENCY_GRAPH_FILE,
)
from tokenflood.analysis import (
    create_summary,
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
