from __future__ import annotations

from typing import Type

import gradio as gr
from plotly import graph_objects as go  # type:ignore[import-untyped]

from tokenflood.visualization_frontend.data import AggregationTrace
from tokenflood.visualization_frontend.graph_style import (
    BASE_COLORS,
    brighten_color,
    aggregation_name_to_color_step,
    aggregation_name_to_line_style,
)
from tokenflood.visualization_frontend.metrics import Metric


def plot_base(trace_groups: list[list[AggregationTrace]]) -> go.Figure:
    fig = go.Figure()
    for i, trace_group in enumerate(trace_groups):
        base_color = BASE_COLORS[i % len(BASE_COLORS)]
        for trace in trace_group:
            color = brighten_color(
                base_color, aggregation_name_to_color_step(trace.aggregation_name)
            )
            style = aggregation_name_to_line_style(trace.aggregation_name)
            fig.add_trace(
                go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    name=trace.aggregation_name,
                    legendgroup=trace.run,
                    legendgrouptitle_text=trace.run,
                    line=dict(color=color, dash=style),
                    hovertemplate="<b>%{fullData.name}</b>: %{y:.2f} ms<extra></extra>",
                )
            )
    fig.update_traces(mode="markers+lines")
    fig.update_layout(
        yaxis_title="latency in ms",
        hovermode="x unified",
        height=900,
        yaxis=dict(rangemode="tozero", ticksuffix=" ms"),
    )
    fig.layout.template = "plotly_dark"
    return fig


def make_observation_latency_plot(
    trace_groups: list[list[AggregationTrace]], metric: Type[Metric]
) -> gr.Plot:
    fig = plot_base(trace_groups)
    fig.update_layout(
        xaxis_title="datetime",
        title=f"{metric.name} over time",
    )
    fig.update_xaxes(tickangle=45)
    return gr.Plot(fig)


def make_run_latency_plot(
    trace_groups: list[list[AggregationTrace]], metric: Type[Metric]
) -> gr.Plot:
    fig = plot_base(trace_groups)
    fig.update_layout(
        xaxis_title="requests per second",
        title=f"{metric.name} across request rates",
        xaxis=dict(ticksuffix=" rps"),
    )
    return gr.Plot(fig)
