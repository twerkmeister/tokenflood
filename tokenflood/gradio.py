import functools
import os
from typing import Dict, List, Tuple

import pandas as pd
import gradio as gr

from tokenflood.analysis import (
    calculate_percentile,
    extend_group_stats,
    get_group_data,
    get_group_stats,
    get_groups,
    mean_int,
)
from tokenflood.constants import (
    ERROR_FILE,
    LLM_REQUESTS_FILE,
    NETWORK_LATENCY_FILE,
    OBSERVATION_SPEC_FILE,
    RUN_SUITE_FILE,
)
from tokenflood.io import (
    is_observation_result_folder,
    is_run_result_folder,
    read_observation_spec,
    read_run_suite,
)
from tokenflood.models.util import numeric


def prepare_observation_data(llm_request_data: pd.DataFrame, all_stats: Dict[str, List[numeric]], stat_names: List[str]) -> pd.DataFrame:
    group_labels = {
        g: get_group_data(llm_request_data, g).reset_index()["datetime"][0][:-9]
        for g in get_groups(llm_request_data)
    }
    group_ids = sorted(group_labels.keys())
    dataframes = []
    for i, name in enumerate(stat_names):
        dataframes.append(
            pd.DataFrame(
                {
                    "datetime": [group_labels[g] for g in group_ids],
                    "latency": [all_stats[g][i] for g in group_ids],
                    "metric": name,
                }
            )
        )

    return pd.concat(dataframes, ignore_index=True)

def prepare_run_data(llm_request_data: pd.DataFrame, all_stats: Dict[str, List[numeric]], stat_names: List[str]) -> pd.DataFrame:
    group_labels = {g: get_group_data(llm_request_data, g).reset_index()["requests_per_second_phase"][0] for g in get_groups(llm_request_data)}
    group_ids = sorted(group_labels.keys())
    dataframes = []
    for i, name in enumerate(stat_names):
        dataframes.append(
            pd.DataFrame({
                "rps": [group_labels[g] for g in group_ids],
                "latency": [all_stats[g][i] for g in group_ids],
                "metric": name
            })
        )

    return pd.concat(dataframes, ignore_index=True)


def get_data(folder: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if is_run_result_folder(folder):
        percentiles = read_run_suite(os.path.join(folder, RUN_SUITE_FILE)).percentiles
    elif is_observation_result_folder(folder):
        percentiles = read_observation_spec(os.path.join(folder, OBSERVATION_SPEC_FILE)).percentiles
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    llm_request_data = pd.read_csv(os.path.join(folder, LLM_REQUESTS_FILE))
    ping_data = pd.read_csv(os.path.join(folder, NETWORK_LATENCY_FILE))

    llm_request_data_aggregations = [mean_int] + [calculate_percentile(p) for p in percentiles]
    request_stats = get_group_stats(llm_request_data, "latency", llm_request_data_aggregations)
    network_stats = get_group_stats(ping_data, "latency", [mean_int])
    all_stats = extend_group_stats(request_stats, network_stats)
    stat_names = ["mean request latency"] + [f"p{p} request latency" for p in percentiles] + ["mean network latency"]
    if is_run_result_folder(folder):
        combined = prepare_run_data(llm_request_data, all_stats, stat_names)
    elif is_observation_result_folder(folder):
        combined = prepare_observation_data(llm_request_data, all_stats, stat_names)
    else:
        combined = pd.DataFrame()

    return combined, llm_request_data, ping_data


def make_observation_latency_plot(data: pd.DataFrame) -> gr.LinePlot:
    return gr.LinePlot(data, title="Latency over time.", x="datetime", y="latency", color="metric", x_title="UTC datetime", y_title="latency in ms", x_label_angle=45, height=500)

def make_run_latency_plot(data: pd.DataFrame) -> gr.LinePlot:
    return gr.LinePlot(data, title="Latency across request rates.", x="rps", y="latency", color="metric", x_title="requests per second", y_title="latency in ms", height=500)


def update_components(results_folder: str, run: str) -> Tuple[gr.LinePlot, gr.DataFrame, gr.DataFrame, gr.DataFrame]:
    run_folder = os.path.join(results_folder, run)
    combined, llm_request_data, ping_data = get_data(run_folder)
    error_data = pd.read_csv(os.path.join(run_folder, ERROR_FILE))
    if is_run_result_folder(run_folder):
        plot = make_run_latency_plot(combined)
    elif is_observation_result_folder(run_folder):
        plot = make_observation_latency_plot(combined)
    else:
        plot = gr.LinePlot()
    return (
        plot,
        gr.DataFrame(llm_request_data, label="llm request data"),
        gr.DataFrame(ping_data, label="ping data"),
        gr.DataFrame(error_data, label="error data")
    )

def load_runs_from_disc(folder: str) -> List[str]:
    runs = sorted(os.listdir(folder), reverse=True)
    runs = [os.path.join(folder, run) for run in runs]
    runs = [run for run in runs if is_observation_result_folder(run) or is_run_result_folder(run)]
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

def visualize_data(results_folder: str):

    runs = load_runs_from_disc(results_folder)
    latest_run = runs[0]
    reload_on_folder_change = functools.partial(update_components, results_folder)
    reload_dropdown_values_from_disc = functools.partial(regularly_update_dropdown, results_folder)
    initial_load = functools.partial(load_state, latest_run)

    with gr.Blocks() as data_visualization:
        timer = gr.Timer(10)
        stored_choice = gr.BrowserState(latest_run)
        dropdown_element = gr.Dropdown(runs, value=latest_run, filterable=True, label="Run Folder")
        line_plot_element, llm_request_data_table, ping_data_table, error_data_table = update_components(results_folder, latest_run)
        dropdown_element.change(reload_on_folder_change, inputs=[dropdown_element], outputs=[line_plot_element, llm_request_data_table, ping_data_table])
        dropdown_element.change(store_choice, inputs=[dropdown_element], outputs=[stored_choice])
        timer.tick(reload_dropdown_values_from_disc, inputs=[dropdown_element], outputs=[dropdown_element])
        data_visualization.load(initial_load, inputs=[stored_choice], outputs=[dropdown_element])

    data_visualization.launch()
