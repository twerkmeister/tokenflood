import os

import pytest
import requests

from tokenflood.visualization_frontend.gradio import (
    visualize_results,
    LOAD_TEST,
    OBSERVATION_TEST,
    get_runs_and_type,
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
    "result_folder, expected_run, expected_starter_type",
    [
        ("empty", [], LOAD_TEST),
        ("only_load_tests", ["load_test_results"], LOAD_TEST),
        ("only_observation_tests", ["observation_results"], OBSERVATION_TEST),
        ("both", ["load_test_results"], LOAD_TEST),
    ],
)
def test_runs_and_type(
    result_folder, expected_run, expected_starter_type, diverse_results_folder
):
    target_folder = os.path.join(diverse_results_folder, result_folder)
    latest_run, _, starter_type = get_runs_and_type(target_folder)
    assert latest_run == expected_run
    assert starter_type == expected_starter_type
