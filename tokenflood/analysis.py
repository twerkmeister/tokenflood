from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

from tokenflood.models.util import numeric

GROUP_ID = "group_id"
REQUESTS_PER_SECOND_FIELD = "requests_per_second_phase"
PERCENTILE_PREFIX = "p"


@dataclass
class AggregationFunc:
    f: Callable[[pd.Series], numeric]
    name: str
    order: float

    def run(self, s: pd.Series) -> numeric:
        return self.f(s)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, AggregationFunc):
            return self.name == other.name
        return False


def get_group_data(data: pd.DataFrame, group: str) -> pd.DataFrame:
    return data[data[GROUP_ID].astype("str") == group]


def get_group_ids(data: pd.DataFrame) -> List[str]:
    return [str(g) for g in pd.unique(data[GROUP_ID])]


def aggregate(
    data: pd.DataFrame, field: str, aggregation_func: AggregationFunc
) -> numeric:
    return aggregation_func.run(data[field])


def mean(data: pd.Series) -> float:
    return float(np.average(data))


Mean = AggregationFunc(mean, "mean", 49.5)


def calculate_percentile(percentile: int) -> AggregationFunc:
    def wrapped(data: pd.Series) -> float:
        return get_percentile_float(list(data), percentile)

    return AggregationFunc(wrapped, f"{PERCENTILE_PREFIX}{percentile}", percentile)


def get_percentile_float(seq: Sequence[numeric], percentile: int) -> float:
    value = 0.0
    if len(seq) == 1:
        value = seq[0]
    elif len(seq) > 1:
        value = float(np.percentile(seq, percentile))
    return value
