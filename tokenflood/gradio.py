import asyncio
from datetime import datetime, timezone
import functools
import logging
import os
import re
from typing import Callable, Dict, List, Tuple, TypeVar

import gradio.routes
import plotly.express as px  # type:ignore[import-untyped]
import pandas as pd
import gradio as gr
from gradio import Blocks

from tokenflood.analysis import (
    calculate_percentile,
    extend_group_stats,
    get_group_data,
    get_group_stats,
    get_groups,
    mean_int,
)
from tokenflood.constants import (
    DEFAULT_PERCENTILES_STR,
    ERROR_FILE,
    LLM_REQUESTS_FILE,
    NETWORK_LATENCY_FILE,
    WARNING_LIMIT_PERCENTAGE,
)
from tokenflood.io import (
    is_observation_result_folder,
    is_run_result_folder,
)
from tokenflood.models.divergence import TokenDivergence
from tokenflood.models.util import numeric

log = logging.getLogger(__name__)


def get_group_labels(
    llm_request_data: pd.DataFrame, label_func: Callable[[pd.DataFrame], str]
) -> Dict[str, str]:
    return {
        g: label_func(get_group_data(llm_request_data, g).reset_index())
        for g in get_groups(llm_request_data)
    }


def get_observation_group_labels(llm_request_data: pd.DataFrame) -> Dict[str, str]:
    return get_group_labels(
        llm_request_data,
        lambda df: datetime.strptime(
            df["datetime"][0][:-9], "%Y-%m-%d_%H-%M-%S"
        ).replace(tzinfo=timezone.utc),
    )


def get_run_group_labels(llm_request_data: pd.DataFrame) -> Dict[str, str]:
    return get_group_labels(
        llm_request_data, lambda df: df["requests_per_second_phase"][0]
    )


def merge_stats(
    llm_request_data: pd.DataFrame,
    all_stats: Dict[str, List[numeric]],
    stat_names: List[str],
    group_label_func: Callable[[pd.DataFrame], Dict[str, str]],
    x_label: str,
    run_name: str,
) -> pd.DataFrame:
    group_labels = group_label_func(llm_request_data)
    group_ids = sorted(group_labels.keys())
    dataframes = []
    for i, name in enumerate(stat_names):
        dataframes.append(
            pd.DataFrame(
                {
                    x_label: [group_labels[g] for g in group_ids],
                    "latency": [all_stats[g][i] for g in group_ids],
                    "metric": f"{run_name}__{name}",
                }
            )
        )

    return pd.concat(dataframes, ignore_index=True)


def make_percentile_labels(percentiles: List[int]) -> List[str]:
    return [f"p{p} request latency" for p in percentiles]


def get_data(
    folder: str, percentiles: List[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not is_run_result_folder(folder) and not is_observation_result_folder(folder):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    llm_request_data = pd.read_csv(os.path.join(folder, LLM_REQUESTS_FILE))
    ping_data = pd.read_csv(os.path.join(folder, NETWORK_LATENCY_FILE))

    llm_request_data_aggregations = [mean_int] + [
        calculate_percentile(p) for p in percentiles
    ]
    request_stats = get_group_stats(
        llm_request_data, "latency", llm_request_data_aggregations
    )
    network_stats = get_group_stats(ping_data, "latency", [mean_int])
    all_stats = extend_group_stats(request_stats, network_stats)
    stat_names = (
        ["mean request latency"]
        + make_percentile_labels(percentiles)
        + ["mean network latency"]
    )
    if is_run_result_folder(folder):
        group_label_func = get_run_group_labels
        x_label = "rps"
    elif is_observation_result_folder(folder):
        group_label_func = get_observation_group_labels
        x_label = "datetime"
    else:
        group_label_func = None
        x_label = None

    if group_label_func is not None and x_label is not None:
        plot_data = merge_stats(
            llm_request_data,
            all_stats,
            stat_names,
            group_label_func,
            x_label,
            os.path.basename(folder),
        )
    else:
        plot_data = pd.DataFrame()

    return plot_data, llm_request_data, ping_data


def make_observation_latency_plot(data: pd.DataFrame) -> gr.Plot:
    fig = px.line(
        data,
        x="datetime",
        y="latency",
        color="metric",
        markers=True,
        title="Latency over time.",
        height=500,
    )
    fig.update_layout(
        xaxis_title="UTC datetime",
        yaxis_title="latency in ms",
    )
    fig.update_xaxes(tickangle=45)
    fig.layout.template = "plotly_dark"
    return gr.Plot(fig)


def make_run_latency_plot(data: pd.DataFrame) -> gr.Plot:
    fig = px.line(
        data,
        x="rps",
        y="latency",
        color="metric",
        markers=True,
        title="Latency across request rates.",
        height=500,
    )
    fig.update_layout(
        xaxis_title="requests per second",
        yaxis_title="latency in ms",
    )
    fig.layout.template = "plotly_dark"
    return gr.Plot(fig)


def get_warning_emoji(relative_error: float) -> str:
    if abs(relative_error) > WARNING_LIMIT_PERCENTAGE:
        return "⚠️"
    return ""


def get_markdown_summary(llm_request_data: pd.DataFrame) -> str:
    if len(llm_request_data) == 0:
        return "Empty data."
    td = TokenDivergence(llm_request_data=llm_request_data)
    return f"""
    ## Token Heuristic Accuracy Stats
    #### Input Tokens {get_warning_emoji(td.relative_input_token_error)}
    On average **{td.mean_expected_input_tokens}** (expected) vs **{td.mean_measured_input_tokens}** (measured) ({td.relative_input_token_error}% error) 
    
    #### Output Tokens {get_warning_emoji(td.relative_output_token_error)}
    On average **{td.mean_expected_output_tokens}** (expected) vs **{td.mean_measured_output_tokens}** (measured) ({td.relative_output_token_error}% error) 
    
    #### Prefix Tokens
    On average **{td.mean_expected_prefix_tokens}** (expected) vs **{td.mean_measured_prefix_tokens}** (measured) ({td.relative_prefix_token_error}% error)
    """


def load_runs_from_disc(folder: str) -> List[str]:
    runs = sorted(os.listdir(folder), reverse=True)
    runs = [os.path.join(folder, run) for run in runs]
    runs = [
        run
        for run in runs
        if is_observation_result_folder(run) or is_run_result_folder(run)
    ]
    runs = [os.path.basename(run) for run in runs]
    return runs


def update_dropdown(results_folder: str) -> gr.Dropdown:
    runs = load_runs_from_disc(results_folder)
    return gr.Dropdown(runs)


T = TypeVar("T")


def id_func(x: T) -> T:
    return x


def load_state(run: str, stored_run: str, stored_percentiles: str) -> Tuple[str, str]:
    percentiles = DEFAULT_PERCENTILES_STR
    if stored_run:
        run = stored_run
    if stored_percentiles:
        percentiles = stored_percentiles
    return run, percentiles


PERCENTILES_SEPARATOR = ","


def percentiles_to_str(percentiles: List[int]) -> str:
    return PERCENTILES_SEPARATOR.join([str(p) for p in percentiles])


def str_to_percentiles(text: str) -> List[int]:
    text = clean_percentiles_input(text)
    splits = text.split(PERCENTILES_SEPARATOR)
    splits = [s for s in splits if s]
    percentiles = [int(s) for s in splits if 0 < int(s) <= 100]
    return sorted(list(set(percentiles)))


def clean_percentiles_input(text: str) -> str:
    """Drop all chars except separator and digits."""
    return re.sub(rf"[^{PERCENTILES_SEPARATOR}0-9]", "", text)


def create_gradio_blocks(results_folder: str) -> Blocks:
    runs = load_runs_from_disc(results_folder)
    latest_run = runs[0]
    reload_dropdown_values_from_disc = functools.partial(
        update_dropdown, results_folder
    )
    initial_load = functools.partial(load_state, latest_run)

    with gr.Blocks() as data_visualization:
        timer = gr.Timer(2)
        stored_run = gr.BrowserState(latest_run)
        stored_percentiles = gr.BrowserState(DEFAULT_PERCENTILES_STR)
        dropdown_element = gr.Dropdown(
            runs,
            value=latest_run,
            multiselect=True,
            filterable=True,
            label="Run Folder",
        )
        percentiles_textbox = gr.Textbox(
            DEFAULT_PERCENTILES_STR,
            label="Percentiles (comma separated, 1-100)",
        )
        dropdown_element.change(
            id_func, inputs=[dropdown_element], outputs=[stored_run]
        )
        dropdown_element.focus(lambda: gr.Timer(active=False), outputs=[timer])
        dropdown_element.blur(lambda: gr.Timer(active=True), outputs=[timer])
        timer.tick(
            reload_dropdown_values_from_disc,
            outputs=[dropdown_element],
        )

        @gr.render(
            inputs=[dropdown_element, percentiles_textbox],
            triggers=[
                dropdown_element.change,
                percentiles_textbox.blur,
                percentiles_textbox.submit,
            ],
        )
        def display_selected_runs(runs: list[str], percentiles_text: str):
            percentiles = str_to_percentiles(percentiles_text)
            is_run_results = len(runs) > 0 and is_run_result_folder(
                os.path.join(results_folder, runs[0])
            )
            is_observation_results = len(runs) > 0 and is_observation_result_folder(
                os.path.join(results_folder, runs[0])
            )
            plot_data_sets = []
            with gr.Tabs(
                selected=0 if len(runs) > 0 else None, render=False
            ) as tab_group:
                for i, run in enumerate(runs):
                    run_folder = os.path.join(results_folder, run)

                    if (
                        is_run_results
                        and not is_run_result_folder(run_folder)
                        or is_observation_results
                        and not is_observation_result_folder(run_folder)
                    ):
                        continue

                    plot_data, llm_request_data, ping_data = get_data(
                        run_folder, percentiles
                    )
                    plot_data_sets.append(plot_data)
                    error_data = pd.read_csv(os.path.join(run_folder, ERROR_FILE))
                    with gr.Tab(run, id=i):
                        gr.Markdown(get_markdown_summary(llm_request_data))
                        gr.DataFrame(
                            llm_request_data,
                            label="llm request data",
                            buttons=["fullscreen", "copy"],
                            show_row_numbers=True,
                            show_search="filter",
                        )
                        gr.DataFrame(
                            ping_data,
                            label="ping data",
                            buttons=["fullscreen", "copy"],
                            show_row_numbers=True,
                            show_search="filter",
                        )
                        gr.DataFrame(
                            error_data,
                            label="error data",
                            buttons=["fullscreen", "copy"],
                            show_row_numbers=True,
                            show_search="filter",
                        )
            if len(plot_data_sets) > 0:
                combined_plot_data = pd.concat(plot_data_sets, ignore_index=True)
                if is_run_results:
                    make_run_latency_plot(combined_plot_data)
                elif is_observation_results:
                    make_observation_latency_plot(combined_plot_data)
                # render tab group after the plot
                tab_group.render()

        percentiles_textbox.blur(
            id_func, inputs=[percentiles_textbox], outputs=[stored_percentiles]
        )
        percentiles_textbox.submit(
            id_func, inputs=[percentiles_textbox], outputs=[stored_percentiles]
        )
        data_visualization.load(
            initial_load,
            inputs=[stored_run, stored_percentiles],
            outputs=[dropdown_element, percentiles_textbox],
        )
    return data_visualization


def visualize_results(
    results_folder: str, keep_running: bool = True, go_to_browser: bool = True
) -> Tuple[gradio.routes.App, str]:
    data_visualization = create_gradio_blocks(results_folder)
    app, url, _ = data_visualization.launch(
        prevent_thread_lock=True, quiet=True, inbrowser=go_to_browser
    )
    log.info(f"Gradio server running at [blue]{url}[/]")
    if keep_running:
        asyncio.get_event_loop().run_forever()

    return app, url
