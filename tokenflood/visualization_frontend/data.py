from __future__ import annotations

import functools
import os.path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, TypeVar, Union, Generic, Type

import pandas as pd

from tokenflood.analysis import (
    get_group_data,
    get_group_ids,
    AggregationFunc,
    aggregate,
)
from tokenflood.models.util import numeric
from tokenflood.visualization_frontend.io import read_dataframe
from tokenflood.visualization_frontend.metrics import Metric

T = TypeVar("T", bound=Union[str, datetime])


@dataclass(frozen=True)
class AggregationTrace(Generic[T]):
    x: list[T]
    y: list[numeric]
    aggregation_name: str
    run: str


@functools.cache
def aggregate_data(
    run_folder: str,
    metric: Type[Metric],
    aggregation_func: AggregationFunc,
    label_func: LabelFunc,
) -> AggregationTrace:
    df = read_dataframe(run_folder, metric.file)
    group_ids = get_group_ids(df)
    x = []
    y = []
    for group_id in group_ids:
        group_data = get_group_data(df, group_id)
        group_label = label_func(group_data)
        x.append(group_label)
        y.append(aggregate(group_data, metric.field_name, aggregation_func))
    return AggregationTrace(x, y, aggregation_func.name, os.path.basename(run_folder))


X = TypeVar("X")
LabelFunc = Callable[[pd.DataFrame], X]


def get_observation_group_label(df: pd.DataFrame) -> datetime:
    date_str = str(df["datetime"].iloc[0][:-9])
    return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)


def get_load_group_label(df: pd.DataFrame) -> str:
    return df["requests_per_second_phase"].iloc[0]
