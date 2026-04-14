from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, TypeVar

import pandas as pd

from tokenflood.analysis import get_group_data, get_group_ids, AggregationFunc, aggregate
from tokenflood.visualization_frontend.io import read_dataframe
from tokenflood.visualization_frontend.metrics import Metric


def aggregate_data(
    folder: str, metric: Metric, aggregation_funcs: list[AggregationFunc], label_func: LabelFunc, x_label: str, metric_suffix: str
) -> pd.DataFrame:
    df = read_dataframe(folder, metric.file)
    group_ids = get_group_ids(df)
    results = []
    for group_id in group_ids:
        group_data = get_group_data(df, group_id)
        group_label = label_func(group_data)
        for f in aggregation_funcs:
            results.append((group_label, aggregate(group_data, metric.field_name, f), f.name + "__" + metric_suffix))

    return pd.DataFrame(results, columns=[x_label, LATENCY_FIELD, METRIC_FIELD])


X = TypeVar("X")
LabelFunc = Callable[[pd.DataFrame], X]


def get_observation_group_label(df: pd.DataFrame) -> datetime:
    date_str = str(df[DATETIME_FIELD].iloc[0][:-9])
    return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)


def get_load_group_label(df: pd.DataFrame) -> str:
    return df["requests_per_second_phase"].iloc[0]


DATETIME_FIELD = "datetime"
REQUESTS_PER_SECOND_FIELD = "rps"
LATENCY_FIELD = "latency"
METRIC_FIELD = "metric"
