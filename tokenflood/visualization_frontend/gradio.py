from __future__ import annotations

import asyncio
import logging
import os
from typing import Tuple, TypeVar, Callable, Type
import gradio.routes
import pandas as pd
import gradio as gr
from gradio import Blocks

from tokenflood import __version__
from tokenflood.analysis import Mean

from tokenflood.constants import (
    DEFAULT_PERCENTILES_STR,
    WARNING_LIMIT_PERCENTAGE,
    RUN_SUITE_FILE,
    OBSERVATION_SPEC_FILE,
    ENDPOINT_SPEC_FILE,
)
from tokenflood.io import get_relative_file_path
from tokenflood.models.divergence import TokenDivergence
from tokenflood.models.util import numeric
from tokenflood.visualization_frontend.data import (
    aggregate_data,
    LabelFunc,
    get_load_group_label,
    get_observation_group_label,
    AggregationTrace,
)
from tokenflood.visualization_frontend.io import (
    get_load_test_runs,
    get_observation_runs,
    get_error_dataframe,
    get_llm_request_dataframe,
    get_network_dataframe,
    get_run_spec_file,
    get_observation_spec_file,
    get_endpoint_spec_file,
)
from tokenflood.visualization_frontend.metrics import (
    RequestLatency,
    NetworkLatency,
    metric_mapping,
    Metric,
)
from tokenflood.visualization_frontend.percentiles import (
    percentiles_to_aggregation_funcs,
)
from tokenflood.visualization_frontend.plots import (
    make_observation_latency_plot,
    make_run_latency_plot,
)

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


def update_runs_for_type(results_folder: str, run_type: str) -> gr.Dropdown:
    runs = load_runs_for_type(results_folder, run_type)
    value = None if len(runs) == 0 else runs[0]
    return gr.Dropdown(runs, value=value)


def get_plot_func(
    run_type: str,
) -> Callable[[list[list[AggregationTrace]], Type[Metric]], gr.Plot]:
    if run_type == LOAD_TEST:
        return make_run_latency_plot
    else:
        return make_observation_latency_plot

def get_label_func(run_type: str) -> LabelFunc:
    if run_type == LOAD_TEST:
        return get_load_group_label
    else:
        return get_observation_group_label

def collect_trace_groups(results_folder: str,
    runs: list[str],
    run_type: str,
    metric: Type[Metric],
    percentiles: str) -> list[list[AggregationTrace]]:
    label_func = get_label_func(run_type)
    aggregation_funcs = sorted(
        [Mean] + percentiles_to_aggregation_funcs(percentiles), key=lambda x: -x.order
    )
    trace_groups: list[list[AggregationTrace]] = []
    for run in runs:
        run_folder = os.path.join(results_folder, run)
        trace_groups.append([])
        for f in aggregation_funcs:
            trace_groups[-1].append(aggregate_data(run_folder, metric, f, label_func))
    return trace_groups

def make_plot(
    results_folder: str,
    runs: list[str],
    run_type: str,
    metric_name: str,
    percentiles: str,
) -> gr.Plot:
    metric = metric_mapping[metric_name]
    trace_groups = collect_trace_groups(results_folder, runs, run_type, metric, percentiles)
    plot_func = get_plot_func(run_type)
    return plot_func(trace_groups, metric)

def make_table(
    results_folder: str,
    runs: list[str],
    run_type: str,
    metric_name: str,
    percentiles: str,
) -> pd.DataFrame:
    metric = metric_mapping[metric_name]
    trace_groups = collect_trace_groups(results_folder, runs, run_type, metric, percentiles)
    rows = []
    for trace_group in trace_groups:
        for trace in trace_group:
            data: dict[str, str | numeric] = {
                "run": trace.run,
                "aggregation": trace.aggregation_name
            }
            for i, x in enumerate(trace.x):
                data[x] = round(trace.y[i], 2)
            rows.append(data)
    return pd.DataFrame(rows)



def make_yaml_code_element(text: str, label: str) -> gr.Code:
    return gr.Code(text, language="yaml", label=label, max_lines=20)


def create_gradio_blocks(results_folder: str) -> Blocks:
    runs = get_load_test_runs(results_folder)
    latest_run = runs[:1]
    title = f"Tokenflood v{__version__}"
    with gr.Blocks(title=title) as blocks:
        timer = gr.Timer(2)
        stored_runs = gr.BrowserState(latest_run, storage_key="runs")
        stored_percentiles = gr.BrowserState(DEFAULT_PERCENTILES_STR, storage_key="percentiles")
        stored_run_type = gr.BrowserState(LOAD_TEST, storage_key="run_type")
        stored_metric = gr.BrowserState(RequestLatency.name, storage_key="metric")
        stored_results_folder = gr.State(results_folder)

        # header - logo and title
        with gr.Row():
            with gr.Column(scale=0, min_width=64):
                gr.Image(
                    get_relative_file_path(__file__, "assets/wave_logo_small.png"),
                    show_label=False,
                    interactive=False,
                    container=False,
                    buttons=[],
                    height=48,
                )
            with gr.Column(scale=1):
                gr.HTML(f"<h1>{title}</h1>")

        # run type and run dropdowns
        with gr.Row():
            with gr.Column(scale=1):
                run_type_dropdown = gr.Dropdown(
                    [LOAD_TEST, OBSERVATION_TEST],
                    value=LOAD_TEST,
                    label="Run type",
                )
            with gr.Column(scale=3):
                runs_dropdown = gr.Dropdown(
                    runs,
                    value=latest_run,
                    multiselect=True,
                    filterable=True,
                    label="Runs",
                )
        # metric and percentile
        with gr.Row():
            with gr.Column(scale=1):
                metric_dropdown = gr.Dropdown(
                    [RequestLatency.name, NetworkLatency.name],
                    value=stored_metric.value,
                    label="Metric",
                )
            with gr.Column(scale=2):
                percentiles_textbox = gr.Textbox(
                    stored_percentiles.value,
                    label="Percentiles (comma separated, 1-100)",
                )

        # dynamic for the selected runs
        @gr.render(
            inputs=[stored_runs, stored_run_type, stored_percentiles, stored_metric],
            triggers=[
                stored_runs.change,
                stored_percentiles.change,
                stored_metric.change,
            ],
            trigger_mode="always_last",
            concurrency_limit=1,
        )
        def display_plot(
            selected_runs: list[str],
            run_type: str,
            percentiles_text: str,
            metric_name: str,
        ):
            if not selected_runs:
                return
            make_plot(
                results_folder, selected_runs, run_type, metric_name, percentiles_text
            )


        # dynamic for the selected runs
        @gr.render(inputs=[stored_runs, stored_run_type],
                   concurrency_limit=1, trigger_mode="always_last")
        def display_run_data(selected_runs: list[str], run_type: str):
            if not selected_runs:
                return
            with gr.Tabs(selected=0):
                for i, run in enumerate(selected_runs):
                    run_folder = os.path.join(results_folder, run)
                    with gr.Tab(run, id=i):
                        llm_request_data = get_llm_request_dataframe(run_folder)
                        gr.HTML("<h2>Token Heuristic Accuracy Stats</h2>")
                        gr.Markdown(get_markdown_summary(llm_request_data))

                        gr.HTML("<h2>Run Files</h2>")
                        with gr.Row():
                            with gr.Column():
                                if run_type == LOAD_TEST:
                                    make_yaml_code_element(
                                        get_run_spec_file(run_folder), RUN_SUITE_FILE
                                    )
                                else:
                                    make_yaml_code_element(
                                        get_observation_spec_file(run_folder),
                                        OBSERVATION_SPEC_FILE,
                                    )
                            with gr.Column():
                                make_yaml_code_element(
                                    get_endpoint_spec_file(run_folder),
                                    ENDPOINT_SPEC_FILE,
                                )

                        gr.HTML("<h2>Raw Data</h2>")
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


        blocks.load(
            args_to_tuple,
            inputs=[stored_runs, stored_run_type, stored_percentiles, stored_metric],
            outputs=[
                runs_dropdown,
                run_type_dropdown,
                percentiles_textbox,
                metric_dropdown,
            ],
        )

        # interactions
        runs_dropdown.change(id_func, inputs=[runs_dropdown], outputs=[stored_runs])
        runs_dropdown.focus(lambda: gr.Timer(active=False), outputs=[timer])
        runs_dropdown.blur(lambda: gr.Timer(active=True), outputs=[timer])
        timer.tick(
            poll_latest_runs,
            inputs=[stored_results_folder, stored_run_type],
            outputs=[runs_dropdown],
        )
        run_type_dropdown.input(
            id_func,
            inputs=[run_type_dropdown],
            outputs=[stored_run_type],
        )
        stored_run_type.change(
            update_runs_for_type,
            inputs=[stored_results_folder, stored_run_type],
            outputs=[runs_dropdown],
        )
        metric_dropdown.change(
            id_func, inputs=[metric_dropdown], outputs=[stored_metric]
        )
        percentiles_textbox.change(
            id_func,
            inputs=percentiles_textbox,
            outputs=stored_percentiles,
            js=create_debounce_js_code("percentiles_textbox_timer", 400),
        )
    return blocks


def visualize_results(
    results_folder: str, keep_running: bool = True, go_to_browser: bool = True
) -> Tuple[gradio.routes.App, str]:
    data_visualization = create_gradio_blocks(results_folder)
    favicon_path = get_relative_file_path(__file__, "assets/wave_logo_small.png")
    app, url, _ = data_visualization.launch(
        prevent_thread_lock=True,
        quiet=True,
        inbrowser=go_to_browser,
        favicon_path=favicon_path,
    )
    log.info(f"Gradio server running at [blue]{url}[/]")
    if keep_running:
        asyncio.get_event_loop().run_forever()

    return app, url
