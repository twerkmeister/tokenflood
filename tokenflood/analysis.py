import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from tokenflood.models.util import numeric

GROUP_ID = "group_id"
REQUESTS_PER_SECOND_FIELD = "requests_per_second_phase"

@dataclass
class AggregationFunc:
    f: Callable[[pd.Series], numeric]
    name: str

    def run(self, s: pd.Series) -> numeric:
        return self.f(s)

def get_group_data(data: pd.DataFrame, group: str) -> pd.DataFrame:
    return data[data[GROUP_ID].astype("str") == group]


def get_group_ids(data: pd.DataFrame) -> List[str]:
    return [str(g) for g in pd.unique(data[GROUP_ID])]


def aggregate(
    data: pd.DataFrame, field: str, aggregation_func: AggregationFunc
) -> numeric:
    return aggregation_func.run(data[field])

def mean_float(data: pd.Series) -> float:
    return round(float(np.average(data)), 2)

MeanFloat = AggregationFunc(mean_float, "mean")

def mean_int(data: pd.Series) -> int:
    return int(np.average(data))

MeanInt = AggregationFunc(mean_int, "mean")

def calculate_percentile(percentile: int) -> AggregationFunc:
    def wrapped(data: pd.Series) -> float:
        return round(get_percentile_float(list(data), percentile), 2)

    return AggregationFunc(wrapped, f"p{percentile}")


def get_percentile_float(seq: Sequence[numeric], percentile: int) -> float:
    value = 0.0
    if len(seq) == 1:
        value = seq[0]
    elif len(seq) > 1:
        value = float(np.percentile(seq, percentile))
    return value
