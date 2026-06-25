from __future__ import annotations

import os.path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, TypeVar, Union, Generic, Type, Sequence

import pandas as pd

from tokenflood.visualization_frontend.aggregation_func import AggregationFunc
from tokenflood.constants import GROUP_ID
from tokenflood.models.util import numeric
from tokenflood.visualization_frontend.io import read_dataframe
from tokenflood.visualization_frontend.metrics import Metric
from tokenflood.visualization_frontend.utils import cache_if_run_data_stayed_the_same

T = TypeVar("T", bound=Union[str, datetime])


@dataclass(frozen=True)
class AggregationTrace(Generic[T]):
    x: list[T]
    y: list[numeric]
    aggregation_name: str
    run: str


@cache_if_run_data_stayed_the_same
def aggregate_data(
    run_folder: str,
    metric: Type[Metric],
    aggregation_funcs: Sequence[AggregationFunc],
) -> list[AggregationTrace]:
    df = read_dataframe(run_folder, metric.file)
    aggregations = {
        aggregation_func.name: pd.NamedAgg(aggregation_func.field, aggregation_func.f)
        for aggregation_func in aggregation_funcs
    }
    aggregated_df = df.groupby(GROUP_ID).agg(**aggregations)
    traces = []
    for func in aggregation_funcs:
        if func.name == "label":
            continue
        traces.append(
            AggregationTrace(
                list(aggregated_df["label"]),
                list(aggregated_df[func.name]),
                func.name,
                os.path.basename(run_folder),
            )
        )
    return traces


X = TypeVar("X")
LabelFunc = Callable[[pd.Series], X]


def get_observation_group_label(s: pd.Series) -> datetime:
    date_str = str(s.iloc[0][:-9])
    return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)


def get_load_group_label(s: pd.Series) -> str:
    return s.iloc[0]
