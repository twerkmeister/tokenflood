import copy
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from tokenflood.models.util import numeric

GROUP_FIELD = "group_id"
REQUESTS_PER_SECOND_FIELD = "requests_per_second_phase"


def get_group_data(data: pd.DataFrame, group: str) -> pd.DataFrame:
    return data[data[GROUP_FIELD].astype("str") == group]


def get_groups(data: pd.DataFrame) -> List[str]:
    return [str(g) for g in pd.unique(data[GROUP_FIELD])]


def aggregate(
    data: pd.DataFrame, field: str, aggregation_func: Callable[[pd.Series], numeric]
) -> numeric:
    return aggregation_func(data[field])


def mean_float(data: pd.Series) -> float:
    return round(float(np.average(data)), 2)


def mean_int(data: pd.Series) -> int:
    return int(np.average(data))


def calculate_percentile(percentile: int) -> Callable[[pd.Series], numeric]:
    def wrapped(data: pd.Series) -> float:
        return round(get_percentile_float(list(data), percentile), 2)

    return wrapped


def get_group_stats(
    data: pd.DataFrame,
    field: str,
    aggregation_funcs: List[Callable[[pd.Series], numeric]],
) -> Dict[str, List[numeric]]:
    groups = get_groups(data)
    result = {}
    for group in groups:
        group_data = get_group_data(data, group)
        result[group] = [aggregate(group_data, field, f) for f in aggregation_funcs]
    return result


def extend_group_stats(
    a: Dict[str, List[numeric]], b: Dict[str, List[numeric]]
) -> Dict[str, List[numeric]]:
    result = copy.deepcopy(a)
    for key in a.keys():
        result[key].extend(b[key])
    return result


def get_percentile_float(seq: Sequence[numeric], percentile: int) -> float:
    value = 0.0
    if len(seq) == 1:
        value = seq[0]
    elif len(seq) > 1:
        value = float(np.percentile(seq, percentile))
    return value
