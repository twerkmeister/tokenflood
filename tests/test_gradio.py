import os

import pandas as pd
import pytest
import requests

from tokenflood.analysis import get_groups
from tokenflood.constants import (
    DEFAULT_PERCENTILES_STR,
    ERROR_FILE,
    LLM_REQUESTS_FILE,
    NETWORK_LATENCY_FILE,
)
from tokenflood.gradio import (
    create_gradio_blocks,
    get_data,
    get_group_labels,
    get_observation_group_labels,
    get_run_group_labels,
    id_func,
    load_runs_from_disc,
    load_state,
    make_observation_latency_plot,
    make_percentile_labels,
    make_run_latency_plot,
    percentiles_to_str,
    str_to_percentiles,
    update_components,
    update_dropdown,
    visualize_results,
)


def test_get_group_labels(observation_results_folder):
    llm_requests_df = pd.read_csv(
        os.path.join(observation_results_folder, LLM_REQUESTS_FILE)
    )
    group_labels = get_group_labels(llm_requests_df, lambda df: str(df["group_id"][0]))
    assert len(group_labels) == 20
    assert all([k == v for k, v in group_labels.items()])


def test_get_observation_group_labels(observation_results_folder):
    llm_requests_df = pd.read_csv(
        os.path.join(observation_results_folder, LLM_REQUESTS_FILE)
    )
    group_labels = get_observation_group_labels(llm_requests_df)
    assert len(group_labels) == 20
    assert group_labels["0"] == "2025-11-18_10-11-57"


def test_get_run_group_labels(run_suite_results_folder):
    llm_requests_df = pd.read_csv(
        os.path.join(run_suite_results_folder, LLM_REQUESTS_FILE)
    )
    group_labels = get_run_group_labels(llm_requests_df)
    assert len(group_labels) == 2
    assert group_labels == {"0": 1.0, "1": 2.0}


def test_make_percentile_labels():
    assert make_percentile_labels([50, 90]) == [
        "p50 request latency",
        "p90 request latency",
    ]


@pytest.mark.parametrize(
    "folder_fixture, x_label, percentiles_str, empty_result",
    [
        ("run_suite_results_folder", "rps", DEFAULT_PERCENTILES_STR, False),
        ("run_suite_results_folder", "rps", "", False),
        ("observation_results_folder", "datetime", DEFAULT_PERCENTILES_STR, False),
        ("unique_temporary_folder", "", DEFAULT_PERCENTILES_STR, True),
    ],
)
def test_get_data(folder_fixture, x_label, percentiles_str, empty_result, request):
    folder = request.getfixturevalue(folder_fixture)
    combined, llm_request_data, ping_data = get_data(
        folder, str_to_percentiles(percentiles_str)
    )
    if empty_result:
        assert combined.empty
        assert llm_request_data.empty
        assert ping_data.empty
    else:
        llm_requests_df = pd.read_csv(os.path.join(folder, LLM_REQUESTS_FILE))
        ping_data_df = pd.read_csv(os.path.join(folder, NETWORK_LATENCY_FILE))
        assert llm_request_data.equals(llm_requests_df)
        assert ping_data.equals(ping_data_df)
        metrics = [
            "mean request latency",
            "mean network latency",
        ] + make_percentile_labels(str_to_percentiles(percentiles_str))
        assert all(combined.columns == [x_label, "latency", "metric"])
        assert set(combined["metric"].unique()) == set(metrics)
        assert len(combined) == len(get_groups(llm_requests_df)) * len(metrics)


def test_make_observation_latency_plot(observation_results_folder):
    combined, llm_request_data, ping_data = get_data(
        observation_results_folder, str_to_percentiles(DEFAULT_PERCENTILES_STR)
    )
    plot = make_observation_latency_plot(combined)
    assert plot.value["type"] == "plotly"


def test_make_run_latency_plot(run_suite_results_folder):
    combined, llm_request_data, ping_data = get_data(
        run_suite_results_folder, str_to_percentiles(DEFAULT_PERCENTILES_STR)
    )
    plot = make_run_latency_plot(combined)
    assert plot.value["type"] == "plotly"


@pytest.mark.parametrize(
    "folder_fixture, empty_result",
    [
        ("run_suite_results_folder", False),
        ("observation_results_folder", False),
        ("unique_temporary_folder", True),
    ],
)
def test_update_components(results_folder, folder_fixture, empty_result, request):
    folder = request.getfixturevalue(folder_fixture)
    run_name = os.path.basename(folder)
    markdown, plot, request_df, ping_df, error_df = update_components(
        results_folder, run_name, DEFAULT_PERCENTILES_STR
    )
    if empty_result:
        assert markdown.value == "Empty data."
        assert plot.value is None
        assert len(request_df.value["data"]) == 0
        assert len(ping_df.value["data"]) == 0
        assert len(error_df.value["data"]) == 0
    else:
        llm_requests_df = pd.read_csv(os.path.join(folder, LLM_REQUESTS_FILE))
        ping_data_df = pd.read_csv(os.path.join(folder, NETWORK_LATENCY_FILE))
        error_data_df = pd.read_csv(os.path.join(folder, ERROR_FILE))
        assert "Empty" not in markdown.value
        assert llm_requests_df.values.tolist() == request_df.value["data"]
        assert ping_data_df.values.tolist() == ping_df.value["data"]
        assert error_data_df.values.tolist() == error_df.value["data"]
        assert plot.value["type"] == "plotly"


def test_load_runs_from_disc(
    results_folder, observation_results_folder, run_suite_results_folder
):
    assert load_runs_from_disc(results_folder) == [
        os.path.basename(run_suite_results_folder),
        os.path.basename(observation_results_folder),
    ]


def test_update_dropdown(results_folder, run_suite_results_folder):
    dropdown = update_dropdown(results_folder)
    assert len(dropdown.choices) == len(load_runs_from_disc(results_folder))
    assert dropdown.value == os.path.basename(run_suite_results_folder)


@pytest.mark.parametrize("val", [1.0, "a", 5, [1, 2, 3]])
def test_id_func(val):
    assert id_func(val) == val


@pytest.mark.parametrize(
    "stored_run, stored_percentiles, latest_run, expected_result",
    [
        ("", "", "abc", ("abc", DEFAULT_PERCENTILES_STR)),
        ("xyz", "50,75,90,99", "abc", ("xyz", "50,75,90,99")),
        ("xyz", "", "", ("xyz", DEFAULT_PERCENTILES_STR)),
    ],
)
def test_load_state(stored_run, stored_percentiles, latest_run, expected_result):
    assert (
        load_state(
            latest_run,
            stored_run,
            stored_percentiles,
        )
        == expected_result
    )


def test_create_gradio_blocks(results_folder):
    data_visualization = create_gradio_blocks(results_folder)
    assert len(data_visualization.blocks) == 11


@pytest.mark.asyncio
async def test_visualize_results(results_folder):
    app, url = visualize_results(results_folder, False, go_to_browser=False)
    response = requests.get(url)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "text, expected_output",
    [
        ("50, 90, 99", [50, 90, 99]),
        ("50,90,99", [50, 90, 99]),
        ("lol50,90,99,abc,!!!", [50, 90, 99]),
        ("50,50,90,99", [50, 90, 99]),
        ("99,90,50,", [50, 90, 99]),
        ("50, 75, 90, 99", [50, 75, 90, 99]),
        ("50, 90, 99, 150", [50, 90, 99]),
        ("0, 50, 90, 99", [50, 90, 99]),
        ("-20, 50, 90, 99", [20, 50, 90, 99]),
        ("", []),
        ("xyz", []),
        (",".join([str(i) for i in range(1, 101)]), list(range(1, 101))),
    ],
)
def test_str_to_percentiles(text, expected_output):
    assert str_to_percentiles(text) == expected_output


def test_percentiles_to_str():
    assert percentiles_to_str([50, 90, 99]) == "50,90,99"
