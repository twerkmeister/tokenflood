from __future__ import annotations

import asyncio
import datetime
import logging
import os
from typing import Tuple, TypeVar, Callable, Type

import gradio.routes
import pandas as pd
import gradio as gr
from gradio import Blocks

from tokenflood import __version__
from tokenflood.visualization_frontend.aggregation_func import AggregationFunc

from tokenflood.constants import (
    DEFAULT_PERCENTILES_STR,
    WARNING_LIMIT_PERCENTAGE,
    LOAD_TEST_SPEC_FILE,
    OBSERVATION_SPEC_FILE,
    ENDPOINT_SPEC_FILE,
)
from tokenflood.io import get_relative_file_path
from tokenflood.models.data.divergence import TokenDivergence
from tokenflood.models.util import numeric
from tokenflood.visualization_frontend.data import (
    aggregate_data,
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
    TimeToFirstToken,
    AverageTimePerOutputToken,
    DecodingLatency,
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
            window.{timer_name} = setTimeout(() => {{
                resolve(text);
            }}, {delay_ms});
        }});
    }}"""


def create_debounce_array_js_code(timer_name: str, delay_ms: int = 500):
    return f"""
    function(selection) {{
        if (window.{timer_name}) clearTimeout(window.{timer_name});
        return new Promise(resolve => {{
            window.{timer_name} = setTimeout(() => {{
                resolve([selection]);
            }}, {delay_ms});
        }});
    }}"""


custom_css = """
.tabs {
    border-width: 1px; /* Light grey border */
    border-color: var(--border-color-primary);
    border-radius: 4px;
}
"""


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


def id_func_list(x: list[str]) -> tuple[list[str], None]:
    return x, None


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
    value = runs[:1]
    return gr.Dropdown(runs, value=value)


def get_plot_func(
    run_type: str,
) -> Callable[[list[list[AggregationTrace]], Type[Metric]], gr.Plot]:
    if run_type == LOAD_TEST:
        return make_run_latency_plot
    else:
        return make_observation_latency_plot


def get_label_func(run_type: str) -> AggregationFunc:
    if run_type == LOAD_TEST:
        return AggregationFunc(
            get_load_group_label, "label", 10000, "requests_per_second_phase"
        )
    else:
        return AggregationFunc(get_observation_group_label, "label", 10000, "datetime")


def collect_trace_groups(
    results_folder: str,
    runs: list[str],
    run_type: str,
    metric: Type[Metric],
    percentiles: str,
) -> list[list[AggregationTrace]]:
    label_func = get_label_func(run_type)
    aggregation_funcs = tuple(
        sorted(
            [
                label_func,
                AggregationFunc(lambda x: x.mean(), "mean", 49.5, metric.field_name),
            ]
            + percentiles_to_aggregation_funcs(percentiles, metric),
            key=lambda x: -x.order,
        )
    )
    trace_groups: list[list[AggregationTrace]] = []
    for run in runs:
        run_folder = os.path.join(results_folder, run)
        trace_groups.append(aggregate_data(run_folder, metric, aggregation_funcs))
    return trace_groups


def make_plot(
    results_folder: str,
    runs: list[str],
    run_type: str,
    metric_name: str,
    percentiles: str,
) -> gr.Plot:
    metric = metric_mapping[metric_name]
    trace_groups = collect_trace_groups(
        results_folder, runs, run_type, metric, percentiles
    )
    plot_func = get_plot_func(run_type)
    return plot_func(trace_groups, metric)


def make_sort_columns(run_type: str) -> Callable[[str], float]:
    def sort_columns_load_test(title: str) -> float:
        if title.endswith(" rps"):
            rps = float(title.split(" ")[0])
            return rps
        else:
            return 0.0

    def sort_columns_observation(title: str) -> float:
        try:
            dtime = datetime.datetime.fromisoformat(title)
            return dtime.timestamp()
        except ValueError:
            return 0.0

    if run_type == LOAD_TEST:
        return sort_columns_load_test
    else:
        return sort_columns_observation


def make_table(
    results_folder: str,
    runs: list[str],
    run_type: str,
    metric_name: str,
    percentiles: str,
) -> pd.DataFrame:
    metric = metric_mapping[metric_name]
    trace_groups = collect_trace_groups(
        results_folder, runs, run_type, metric, percentiles
    )
    rows = []
    for trace_group in trace_groups:
        for trace in trace_group:
            data: dict[str, str | numeric] = {
                "run": trace.run,
                "aggregation": trace.aggregation_name,
                "metric": metric_name,
            }
            for i, x in enumerate(trace.x):
                if run_type == LOAD_TEST:
                    data[str(x) + " rps"] = str(round(trace.y[i], 2)) + " ms"
                elif run_type == OBSERVATION_TEST:
                    data[str(x)] = str(round(trace.y[i], 2)) + " ms"
                else:
                    data[x] = round(trace.y[i])
            rows.append(data)
    return pd.DataFrame(rows).sort_index(
        axis=1, key=lambda x: x.map(make_sort_columns(run_type))
    )


def update_data(
    results_folder: str,
    runs: list[str],
    run_type: str,
    metric_name: str,
    percentiles: str,
) -> tuple[gr.Plot, gr.DataFrame]:
    return (
        make_plot(results_folder, runs, run_type, metric_name, percentiles),
        gr.DataFrame(
            make_table(results_folder, runs, run_type, metric_name, percentiles),
            visible=False,
        ),
    )


def make_frame_visible(data) -> gr.DataFrame:
    return gr.DataFrame(data, visible=True)


def make_yaml_code_element(text: str, label: str) -> gr.Code:
    return gr.Code(text, language="yaml", label=label, max_lines=20)


def on_select(evt: gr.SelectData):
    return evt.index


def get_runs_and_type(results_folder) -> tuple[list[str], list[str], str]:
    load_tests = get_load_test_runs(results_folder)
    latest_runs = load_tests[:1]
    runs = load_tests
    run_type = LOAD_TEST
    observation_tests = get_observation_runs(results_folder)
    if len(load_tests) == 0 and len(observation_tests) > 0:
        latest_runs = observation_tests[:1]
        runs = observation_tests
        run_type = OBSERVATION_TEST
    return latest_runs, runs, run_type

custom_js = """
    const waitForElement = async (selector, interval = 500) => {
      while (true) {
        const element = document.querySelector(selector);
        if (element) return element;
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    };
    
    const initPlotlySelectionHook = () => {
      waitForElement('#main_plot').then(el => {
        const attach = () => {
          const gd = document.querySelector('#main_plot .js-plotly-plot');
          if (!gd || gd._selHooked) return;
          gd._selHooked = true;
          gd.original_names = gd.data.map(({name}) => name);
          gd.isUpdatingPlotly = false;
          // console.log("attached");
          gd.on('plotly_selected', (ev) => {
            if (gd.isUpdatingPlotly) return;
            // console.log("selected event fired");
            // console.log(ev);
            gd.isUpdatingPlotly = true;
            if (ev && ev.points) {
              const grouped = Object.groupBy(ev.points, point => point.curveNumber);
              // console.log(grouped);
              const averages = Object.entries(grouped).map(([curveNumber, points]) => {
                const total = points.reduce((sum, item) => sum + item.y, 0);
                const average = Math.round((total / points.length) * 100)/100;
                return {
                  curveNumber,
                  average,
                  numberOfPoints: points.length
                }
              });
              // console.log(averages);
              averages.forEach(({curveNumber, average, numberOfPoints}) => {
                window.Plotly.restyle(gd, { name: gd.original_names[curveNumber] + " (selection average based on " + numberOfPoints.toString() + " points: " + average.toString() + "ms)" }, [curveNumber]);    
              });
              gd.isUpdatingPlotly = false;
            } else {
              window.Plotly.restyle(gd, { name: gd.original_names }, Array(gd.original_names.length).keys()).then(() => {
                gd.isUpdatingPlotly.false;
              })
            }
          });
        };
        attach();
        new MutationObserver(attach).observe(el, { childList: true, subtree: true });
      });
    }
    
    initPlotlySelectionHook();
    
"""


def create_gradio_blocks(results_folder: str) -> Blocks:
    latest_runs, all_runs, starter_run_type = get_runs_and_type(results_folder)
    title = f"Tokenflood v{__version__}"
    with gr.Blocks(title=title, analytics_enabled=False) as blocks:
        timer = gr.Timer(2)
        stored_percentiles = gr.State(DEFAULT_PERCENTILES_STR)
        stored_results_folder = gr.State(results_folder)
        stored_runs = gr.State(latest_runs)
        dummy_state = gr.State(
            None
        )  # needed for debounce js of runs dropdown to make array return possible

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
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                run_type_dropdown = gr.Dropdown(
                    [LOAD_TEST, OBSERVATION_TEST],
                    value=starter_run_type,
                    label="Run type",
                )
            with gr.Column(scale=3):
                runs_dropdown = gr.Dropdown(
                    all_runs,
                    value=latest_runs,
                    multiselect=True,
                    filterable=True,
                    interactive=True,
                    label="Runs",
                )
        # metric and percentile
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                metric_dropdown = gr.Dropdown(
                    [
                        RequestLatency.name,
                        TimeToFirstToken.name,
                        DecodingLatency.name,
                        AverageTimePerOutputToken.name,
                        NetworkLatency.name,
                    ],
                    value=RequestLatency.name,
                    label="Metric",
                    info=RequestLatency.explanation,
                )
            with gr.Column(scale=2):
                percentiles_textbox = gr.Textbox(
                    stored_percentiles.value,
                    label="Percentiles (comma separated, 1-100)",
                    interactive=True,
                )

        data_plot = make_plot(
            results_folder,
            latest_runs,
            starter_run_type,
            RequestLatency.name,
            DEFAULT_PERCENTILES_STR,
        )
        data_table = gr.DataFrame(
            make_table(
                results_folder,
                latest_runs,
                starter_run_type,
                RequestLatency.name,
                DEFAULT_PERCENTILES_STR,
            ),
            label="tabulated data",
            interactive=False,
        )
        # triggering visibility of the dataframe to force rerender and make all data lines show up
        gr.on(
            [stored_runs.change, stored_percentiles.change, metric_dropdown.change],
            update_data,
            inputs=[
                stored_results_folder,
                stored_runs,
                run_type_dropdown,
                metric_dropdown,
                stored_percentiles,
            ],
            outputs=[data_plot, data_table],
            trigger_mode="always_last",
            concurrency_limit=1,
        ).then(make_frame_visible, inputs=[data_table], outputs=[data_table])

        @gr.render(
            inputs=[stored_runs, run_type_dropdown],
            triggers=[stored_runs.change, blocks.load],
            concurrency_limit=1,
            trigger_mode="always_last",
        )
        def render_tabs(
            selected_runs: list[str],
            run_type: str,
        ):
            with gr.Tabs():
                for i, run in enumerate(selected_runs):
                    run_folder = os.path.join(results_folder, run)
                    with gr.Tab(run, id=i):
                        llm_request_data = get_llm_request_dataframe(run_folder)
                        with gr.Accordion("Token Heuristic Accuracy Stats"):
                            gr.Markdown(get_markdown_summary(llm_request_data))
                        with gr.Accordion("Run Files", open=False):
                            with gr.Row():
                                with gr.Column():
                                    if run_type == LOAD_TEST:
                                        make_yaml_code_element(
                                            get_run_spec_file(run_folder),
                                            LOAD_TEST_SPEC_FILE,
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
                        with gr.Accordion("Raw Request Data", open=False):
                            gr.DataFrame(
                                llm_request_data,
                                label="llm request data",
                                buttons=["fullscreen", "copy"],
                                show_row_numbers=True,
                                show_search="filter",
                            )
                        with gr.Accordion("Raw ping Data", open=False):
                            gr.DataFrame(
                                get_network_dataframe(run_folder),
                                label="Ping data",
                                buttons=["fullscreen", "copy"],
                                show_row_numbers=True,
                                show_search="filter",
                            )
                        with gr.Accordion("Error data", open=False):
                            gr.DataFrame(
                                get_error_dataframe(run_folder),
                                label="error data",
                                buttons=["fullscreen", "copy"],
                                show_row_numbers=True,
                                show_search="filter",
                            )

        runs_dropdown.focus(lambda: gr.Timer(active=False), outputs=[timer])
        runs_dropdown.blur(lambda: gr.Timer(active=True), outputs=[timer])
        timer.tick(
            poll_latest_runs,
            inputs=[stored_results_folder, run_type_dropdown],
            outputs=[runs_dropdown],
        )
        run_type_dropdown.change(
            update_runs_for_type,
            inputs=[stored_results_folder, run_type_dropdown],
            outputs=[runs_dropdown],
        )
        runs_dropdown.change(
            id_func_list,
            inputs=[runs_dropdown],
            outputs=[stored_runs, dummy_state],
            js=create_debounce_array_js_code("runs_dropdown_debounce", 500),
        )
        percentiles_textbox.change(
            id_func,
            inputs=percentiles_textbox,
            outputs=stored_percentiles,
            js=create_debounce_js_code("percentiles_textbox_timer", 400),
        )
        metric_dropdown.change(
            lambda m: gr.Dropdown(info=metric_mapping[m].explanation),
            inputs=metric_dropdown,
            outputs=metric_dropdown,
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
        css=custom_css,
        js=custom_js,
    )
    log.info(f"Gradio server running at [blue]{url}[/]")
    if keep_running:
        asyncio.get_event_loop().run_forever()

    return app, url
