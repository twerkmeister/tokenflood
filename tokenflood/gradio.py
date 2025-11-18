import functools
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from tokenflood.analysis import get_group_data, get_groups, get_percentile_float
from tokenflood.io import read_observation_spec
from tokenflood.models.run_summary import LoadResult


def get_observation_data(folder: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    observation_spec = read_observation_spec(os.path.join(folder, "observation_spec.yml"))
    llm_request_data = pd.read_csv(os.path.join(folder, "llm_requests.csv"))
    ping_data = pd.read_csv(os.path.join(folder, "network_latency.csv"))

    groups = get_groups(llm_request_data)
    load_results = []
    group_labels = []
    for group in groups:
        group_llm_request_data = get_group_data(llm_request_data, group)
        group_labels.append(group_llm_request_data.reset_index()["datetime"][0][:-9])
        group_ping_data = get_group_data(ping_data, group)
        percentiles = {}
        for percentile in observation_spec.percentiles:
            percentiles[f"p{percentile}"] = round(
                get_percentile_float(
                    list(group_llm_request_data["latency"]), percentile
                ),
                2,
            )
        load_results.append(
            LoadResult(
                requests_per_second=float(group),
                mean_request_latency=round(
                    float(np.average(group_llm_request_data["latency"])), 2
                ),
                mean_network_latency=round(
                    float(np.average(group_ping_data["latency"])), 2
                ),
                percentile_latency=percentiles,
            )
        )
    dfs = [
        pd.DataFrame(
            {
                "datetime": group_labels,
                "latency": [lr.mean_request_latency for lr in load_results],
                "metric": "mean_request_latency",
            }
        ),
        pd.DataFrame(
            {
                "datetime": group_labels,
                "latency": [lr.mean_network_latency for lr in load_results],
                "metric": "mean_network_latency",
            }
        ),
    ]
    for percentile in load_results[0].percentile_latency:
        dfs.append(
            pd.DataFrame(
                {
                    "datetime": group_labels,
                    "latency": [
                        lr.percentile_latency[percentile] for lr in load_results
                    ],
                    "metric": f"{percentile} latency",
                }
            )
        )

    combined = pd.concat(dfs, ignore_index=True)

    return combined, llm_request_data, ping_data

def update_gradio_components(results_folder: str, run_folder: str) -> Tuple[gr.LinePlot, gr.DataFrame, gr.DataFrame]:
    combined, llm_request_data, ping_data = get_observation_data(os.path.join(results_folder, run_folder))
    return (
        gr.LinePlot(combined, title="Latency over time.", x="datetime", y="latency", color="metric", x_title="UTC datetime", y_title="latency in ms", x_label_angle=45, height=500),
        gr.DataFrame(llm_request_data, label="llm request data"),
        gr.DataFrame(ping_data, label="ping data")
    )

def load_runs_from_disc(results_folder: str) -> List[str]:
    runs = sorted(os.listdir(results_folder), reverse=True)
    runs = [run for run in runs if os.path.isfile(os.path.join(results_folder, run, "observation_spec.yml"))]
    return runs

def regularly_update_dropdown(results_folder: str, current_run: str) -> gr.Dropdown:
    runs = load_runs_from_disc(results_folder)
    return gr.Dropdown(runs, value=current_run, filterable=True, label="Run Folder")

def store_choice(choice: str) -> str:
    return choice

def load_state(latest_run: str, state: str) -> str:
    if state:
        return state
    return latest_run

def visualize_observation_data(results_folder: str):

    runs = load_runs_from_disc(results_folder)
    latest_run = runs[0]
    combined, llm_request_data, ping_data = get_observation_data(os.path.join(results_folder, latest_run))
    reload_on_folder_change = functools.partial(update_gradio_components, results_folder)
    reload_dropdown_values_from_disc = functools.partial(regularly_update_dropdown, results_folder)
    initial_load = functools.partial(load_state, latest_run)

    with gr.Blocks() as observation_data_visualization:
        timer = gr.Timer(10)
        stored_choice = gr.BrowserState(latest_run)
        dropdown_element = gr.Dropdown(runs, value=latest_run, filterable=True, label="Run Folder")
        line_plot_element = gr.LinePlot(combined, title="Latency over time.", x="datetime", y="latency", color="metric", x_title="UTC datetime", y_title="latency in ms", x_label_angle=45, height=500)
        llm_request_data_table = gr.DataFrame(llm_request_data, label="llm request data")
        ping_data_table = gr.DataFrame(ping_data, label="ping data")
        dropdown_element.change(reload_on_folder_change, inputs=[dropdown_element], outputs=[line_plot_element, llm_request_data_table, ping_data_table])
        dropdown_element.change(store_choice, inputs=[dropdown_element], outputs=[stored_choice])
        timer.tick(reload_dropdown_values_from_disc, inputs=[dropdown_element], outputs=[dropdown_element])
        observation_data_visualization.load(initial_load, inputs=[stored_choice], outputs=[dropdown_element])

    observation_data_visualization.launch()
