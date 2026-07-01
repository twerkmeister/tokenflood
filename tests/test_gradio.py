import os

import pytest
import requests

from tokenflood.constants import DEFAULT_PERCENTILES_STR
from tokenflood.visualization_frontend.gradio import (
    visualize_results,
    LOAD_TEST,
    OBSERVATION_TEST,
    initialize_run_type_from_url,
    initialize_runs_from_url,
    initialize_metric_from_url,
    initialize_percentiles_from_url,
    RUN_TYPE_QUERY_PARAM,
    METRIC_QUERY_PARAM,
    RUNS_QUERY_PARAM,
)
from tokenflood.visualization_frontend.metrics import (
    RequestLatency,
    TimeToFirstToken,
    DecodingLatency,
    AverageTimePerOutputToken,
    NetworkLatency,
)
from tokenflood.visualization_frontend.percentiles import (
    percentiles_to_str,
    str_to_percentiles,
)


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


@pytest.mark.parametrize(
    "result_folder, params, expected_starter_type",
    [
        ("empty", {}, LOAD_TEST),
        ("empty", {RUN_TYPE_QUERY_PARAM: LOAD_TEST}, LOAD_TEST),
        ("empty", {RUN_TYPE_QUERY_PARAM: OBSERVATION_TEST}, OBSERVATION_TEST),
        ("empty", {RUN_TYPE_QUERY_PARAM: "nonexistent"}, LOAD_TEST),
        ("empty", {RUN_TYPE_QUERY_PARAM: ""}, LOAD_TEST),
        ("empty", {RUN_TYPE_QUERY_PARAM: None}, LOAD_TEST),
        ("only_load_tests", {}, LOAD_TEST),
        ("only_load_tests", {RUN_TYPE_QUERY_PARAM: LOAD_TEST}, LOAD_TEST),
        ("only_load_tests", {RUN_TYPE_QUERY_PARAM: OBSERVATION_TEST}, OBSERVATION_TEST),
        ("only_load_tests", {RUN_TYPE_QUERY_PARAM: "nonexistent"}, LOAD_TEST),
        ("only_observation_tests", {}, OBSERVATION_TEST),
        ("only_observation_tests", {RUN_TYPE_QUERY_PARAM: LOAD_TEST}, LOAD_TEST),
        (
            "only_observation_tests",
            {RUN_TYPE_QUERY_PARAM: OBSERVATION_TEST},
            OBSERVATION_TEST,
        ),
        (
            "only_observation_tests",
            {RUN_TYPE_QUERY_PARAM: "nonexistent"},
            OBSERVATION_TEST,
        ),
        ("both", {}, LOAD_TEST),
        ("both", {RUN_TYPE_QUERY_PARAM: LOAD_TEST}, LOAD_TEST),
        ("both", {RUN_TYPE_QUERY_PARAM: OBSERVATION_TEST}, OBSERVATION_TEST),
        ("both", {RUN_TYPE_QUERY_PARAM: "nonexistent"}, LOAD_TEST),
    ],
)
def test_initialize_run_type_from_url(
    result_folder, params, expected_starter_type, diverse_results_folder
):
    target_folder = os.path.join(diverse_results_folder, result_folder)
    assert expected_starter_type == initialize_run_type_from_url(target_folder, params)


@pytest.mark.parametrize(
    "result_folder, run_type, params, expected_runs",
    [
        # Valid cases
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "load1"}, ["load1"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "load1, load2"}, ["load1", "load2"]),
        (
            "both",
            OBSERVATION_TEST,
            {RUNS_QUERY_PARAM: "observation1"},
            ["observation1"],
        ),
        # order preserved
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "load2, load1"}, ["load2", "load1"]),
        # empty folder
        ("empty", LOAD_TEST, {}, []),
        ("empty", OBSERVATION_TEST, {}, []),
        # opposite run type in folder (system sticks to specified run type)
        ("only_load_tests", OBSERVATION_TEST, {}, []),
        ("only_observation_tests", LOAD_TEST, {}, []),
        # Fallback to first available if no valid runs provided
        ("both", LOAD_TEST, {}, ["load2"]),
        ("both", OBSERVATION_TEST, {}, ["observation1"]),
        # Path traversal attempts (should be stripped or ignored)
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "../secret"}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "/etc/passwd"}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "loadx/../../etc/passwd"}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "loadx/../.htpasswd"}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: ""}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "/"}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "// > /dev/null"}, ["load2"]),
        # Special characters and empty parts
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "load1, , ,"}, ["load1"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "."}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: ".."}, ["load2"]),
        # Wrong directory names or types
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "nonexistent"}, ["load2"]),
        ("both", LOAD_TEST, {RUNS_QUERY_PARAM: "obs_1"}, ["load2"]),
        ("both", OBSERVATION_TEST, {RUNS_QUERY_PARAM: "load1"}, ["observation1"]),
    ],
)
def test_initialize_runs_from_url(
    result_folder, run_type, params, expected_runs, diverse_results_folder
):
    target_folder = os.path.join(diverse_results_folder, result_folder)
    runs, _ = initialize_runs_from_url(target_folder, run_type, params)
    assert runs == expected_runs


@pytest.mark.parametrize(
    "params, expected_metric",
    [
        ({}, RequestLatency.name),
        ({METRIC_QUERY_PARAM: TimeToFirstToken.name}, TimeToFirstToken.name),
        ({METRIC_QUERY_PARAM: DecodingLatency.name}, DecodingLatency.name),
        (
            {METRIC_QUERY_PARAM: AverageTimePerOutputToken.name},
            AverageTimePerOutputToken.name,
        ),
        ({METRIC_QUERY_PARAM: NetworkLatency.name}, NetworkLatency.name),
        ({METRIC_QUERY_PARAM: "NonExistentMetric"}, RequestLatency.name),
        ({METRIC_QUERY_PARAM: ""}, RequestLatency.name),
    ],
)
def test_initialize_metric_from_url(params, expected_metric):
    assert initialize_metric_from_url(params) == expected_metric


@pytest.mark.parametrize(
    "params, expected_percentiles",
    [
        ({}, DEFAULT_PERCENTILES_STR),
        ({"percentiles": "50, 95"}, "50,95"),
        ({"percentiles": "lol,90, 99, 150"}, "90,99"),
        ({"percentiles": ""}, ""),
        ({"percentiles": "30,5,10"}, "5,10,30"),
        ({"percentiles": "test,5,6,1,bad"}, "1,5,6"),
        ({"percentiles": "-1,95,120"}, "1,95"),
        ({"percentiles": "abc1,95,120"}, "1,95"),
    ],
)
def test_initialize_percentiles_from_url(params, expected_percentiles):
    assert initialize_percentiles_from_url(params) == expected_percentiles
