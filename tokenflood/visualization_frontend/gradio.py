from __future__ import annotations

import asyncio
import logging
import os
from typing import Tuple, TypeVar, Callable
import gradio.routes
import plotly.express as px  # type:ignore[import-untyped]
import pandas as pd
import gradio as gr
from gradio import Blocks

from tokenflood import __version__
from tokenflood.analysis import MeanInt

from tokenflood.constants import (
    DEFAULT_PERCENTILES_STR,
    WARNING_LIMIT_PERCENTAGE,
)
from tokenflood.models.divergence import TokenDivergence
from tokenflood.visualization_frontend.data import aggregate_data, LabelFunc, get_load_group_label, \
    get_observation_group_label, DATETIME_FIELD, REQUESTS_PER_SECOND_FIELD
from tokenflood.visualization_frontend.io import get_load_test_runs, get_observation_runs, get_error_dataframe, \
    get_llm_request_dataframe, get_network_dataframe
from tokenflood.visualization_frontend.metrics import RequestLatency, NetworkLatency, metric_mapping
from tokenflood.visualization_frontend.percentiles import percentiles_to_aggregation_funcs
from tokenflood.visualization_frontend.plots import make_observation_latency_plot, make_run_latency_plot

log = logging.getLogger(__name__)

LOAD_TEST = "load-test"
OBSERVATION_TEST = "observation"


def create_debounce_js_code(timer_name: str, delay_ms: int = 500):
    return f"""
    function(text) {{
        if (window.{timer_name}) clearTimeout(window.{timer_name});
        return new Promise(resolve => {{
            window.{timer_name} = setTimeout(() => resolve(text), {delay_ms});
        }});
    }}"""


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


T = TypeVar("T")

def id_func(x: T) -> T:
    return x

def args_to_tuple(*args):
    return args


def load_runs_for_type(results_folder: str, run_type: str) -> list[str]:
    if run_type == LOAD_TEST:
        return get_load_test_runs(results_folder)
    else:
        return get_observation_runs(results_folder)

def poll_latest_runs(results_folder: str, run_type: str) -> gr.Dropdown:
    return gr.Dropdown(load_runs_for_type(results_folder, run_type))

def update_run_type(results_folder: str, run_type: str) -> gr.Dropdown:
    runs = load_runs_for_type(results_folder, run_type)
    value = None if len(runs) == 0 else runs[0]
    return gr.Dropdown(runs, value=value)

def get_plot_variables(run_type: str) -> tuple[str, LabelFunc, Callable[[pd.DataFrame], gr.Plot]]:
    if run_type == LOAD_TEST:
        return REQUESTS_PER_SECOND_FIELD, get_load_group_label, make_run_latency_plot
    else:
        return DATETIME_FIELD, get_observation_group_label, make_observation_latency_plot

def make_plot(results_folder: str, runs: list[str], run_type: str, metric_name: str, percentiles: str) -> gr.Plot:
    metric = metric_mapping[metric_name]
    dfs = []
    x_label, label_func, plot_func = get_plot_variables(run_type)
    aggregation_funcs = [MeanInt] + percentiles_to_aggregation_funcs(percentiles)
    for run in runs:
        run_folder = os.path.join(results_folder, run)
        dfs.append(aggregate_data(run_folder, metric, aggregation_funcs, label_func, x_label, run))
    df = pd.concat(dfs, ignore_index=True)
    return plot_func(df)

def create_gradio_blocks(results_folder: str) -> Blocks:
    runs = get_load_test_runs(results_folder)
    latest_run = None if len(runs) == 0 else runs[:1]
    title = f"Tokenflood v{__version__}"
    with gr.Blocks() as blocks:
        gr.HTML(f"<h1>{title}</h1>")
        timer = gr.Timer(2)
        stored_runs = gr.BrowserState(latest_run)
        stored_percentiles = gr.BrowserState(DEFAULT_PERCENTILES_STR)
        stored_run_type = gr.BrowserState(LOAD_TEST)
        stored_metric = gr.BrowserState(RequestLatency.__name__)
        stored_results_folder = gr.State(results_folder)

        with gr.Row():
            with gr.Column(scale=1):
                run_type_dropdown = gr.Dropdown(
                    [LOAD_TEST, OBSERVATION_TEST],
                    value=stored_run_type.value,
                    label="Run type",
                    interactive=True,
                )
            with gr.Column(scale=3):
                runs_dropdown = gr.Dropdown(
                    runs,
                    value=stored_runs.value,
                    multiselect=True,
                    filterable=True,
                    label="Runs",
                    interactive=True,
                )

            runs_dropdown.change(
                id_func, inputs=[runs_dropdown], outputs=[stored_runs]
            )
            runs_dropdown.focus(lambda: gr.Timer(active=False), outputs=[timer])
            runs_dropdown.blur(lambda: gr.Timer(active=True), outputs=[timer])
            timer.tick(
                poll_latest_runs,
                inputs=[stored_results_folder, stored_run_type],
                outputs=[runs_dropdown],
            )
            run_type_dropdown.change(
                update_run_type,
                inputs=[stored_results_folder, stored_run_type],
                outputs=[runs_dropdown],
            )
        with gr.Row():
            with gr.Column(scale=1):
                metric_dropdown = gr.Dropdown(
                    [RequestLatency.__name__, NetworkLatency.__name__],
                    stored_metric.value,
                    label="Metric"
                )
                metric_dropdown.change(id_func, inputs=[metric_dropdown], outputs=[stored_metric])
            with gr.Column(scale=2):
                percentiles_textbox = gr.Textbox(
                    stored_percentiles.value,
                    label="Percentiles (comma separated, 1-100)",
                )
                percentiles_textbox.change(
                    id_func,
                    inputs=percentiles_textbox,
                    outputs=stored_percentiles,
                    js=create_debounce_js_code("percentiles_textbox_timer", 400),
                )

        @gr.render(
            inputs=[stored_runs, stored_run_type, stored_percentiles, stored_metric],
        )
        def display_selected_runs(selected_runs: list[str], run_type: str, percentiles_text: str, metric_name: str):
            if not selected_runs:
                return
            make_plot(results_folder, selected_runs, run_type, metric_name, percentiles_text)
            with gr.Tabs(selected=0 if len(selected_runs) > 0 else None):
                for i, run in enumerate(selected_runs):
                    run_folder = os.path.join(results_folder, run)
                    with gr.Tab(run, id=i):
                        llm_request_data = get_llm_request_dataframe(run_folder)
                        gr.Markdown(get_markdown_summary(llm_request_data))
                        gr.DataFrame(
                            llm_request_data,
                            label="llm request data",
                            buttons=["fullscreen", "copy"],
                            show_row_numbers=True,
                            show_search="filter",
                        )
                        gr.DataFrame(
                            get_network_dataframe(run_folder),
                            label="ping data",
                            buttons=["fullscreen", "copy"],
                            show_row_numbers=True,
                            show_search="filter",
                        )
                        gr.DataFrame(
                            get_error_dataframe(run_folder),
                            label="error data",
                            buttons=["fullscreen", "copy"],
                            show_row_numbers=True,
                            show_search="filter",
                        )
        blocks.load(args_to_tuple,
                    inputs=[stored_runs, stored_run_type, stored_percentiles, stored_metric],
                    outputs=[runs_dropdown, run_type_dropdown, percentiles_textbox, metric_dropdown])
    return blocks


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
