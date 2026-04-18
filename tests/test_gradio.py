import pytest
import requests

from tokenflood.visualization_frontend.gradio import (
    create_gradio_blocks,
    visualize_results,
)
from tokenflood.visualization_frontend.percentiles import (
    percentiles_to_str,
    str_to_percentiles,
)


def test_create_gradio_blocks(results_folder):
    data_visualization = create_gradio_blocks(results_folder)
    assert len(data_visualization.blocks) == 8


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
