from __future__ import annotations

import gradio as gr
import pandas as pd
from plotly import express as px

from tokenflood.visualization_frontend.data import DATETIME_FIELD, REQUESTS_PER_SECOND_FIELD, LATENCY_FIELD, \
    METRIC_FIELD
from tokenflood.visualization_frontend.graph_style import assign_metric_colors, assign_metric_line_style


def make_observation_latency_plot(data: pd.DataFrame) -> gr.Plot:
    data = data.sort_values(DATETIME_FIELD)
    metrics = get_unique_metrics(data)
    fig = px.line(
        data,
        x=DATETIME_FIELD,
        y=LATENCY_FIELD,
        color=METRIC_FIELD,
        line_dash=METRIC_FIELD,
        color_discrete_map=assign_metric_colors(metrics),
        line_dash_map=assign_metric_line_style(metrics),
        markers=True,
        title="Latency over time.",
        height=900,
    )
    fig.update_layout(
        xaxis_title="UTC datetime",
        yaxis_title="latency in ms",
    )
    fig.update_xaxes(tickangle=45)
    fig.layout.template = "plotly_dark"
    return gr.Plot(fig)


def make_run_latency_plot(data: pd.DataFrame) -> gr.Plot:
    metrics = get_unique_metrics(data)
    fig = px.line(
        data,
        x=REQUESTS_PER_SECOND_FIELD,
        y=LATENCY_FIELD,
        color=METRIC_FIELD,
        line_dash=METRIC_FIELD,
        color_discrete_map=assign_metric_colors(metrics),
        line_dash_map=assign_metric_line_style(metrics),
        markers=True,
        title="Latency across request rates.",
        height=900,
    )
    fig.update_layout(
        xaxis_title="requests per second",
        yaxis_title="latency in ms",
    )
    fig.layout.template = "plotly_dark"
    return gr.Plot(fig)


def get_unique_metrics(data: pd.DataFrame) -> list[str]:
    metrics = data[METRIC_FIELD].unique()
    return [str(m) for m in metrics]
