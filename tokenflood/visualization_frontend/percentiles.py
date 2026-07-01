from __future__ import annotations

import re
from typing import List, Type

from tokenflood.visualization_frontend.aggregation_func import AggregationFunc
from tokenflood.visualization_frontend.metrics import Metric

PERCENTILES_SEPARATOR = ","


def percentiles_to_str(percentiles: List[int]) -> str:
    return PERCENTILES_SEPARATOR.join([str(p) for p in percentiles])


def str_to_percentiles(text: str) -> List[int]:
    text = clean_percentiles_input(text)
    splits = text.split(PERCENTILES_SEPARATOR)
    splits = [s for s in splits if s]
    percentiles = [int(s) for s in splits if 0 < int(s) <= 100]
    return sorted(list(set(percentiles)))


def clean_percentiles_input(text: str) -> str:
    """Drop all chars except separator and digits."""
    return re.sub(rf"[^{PERCENTILES_SEPARATOR}0-9]", "", text)


def percentiles_to_aggregation_funcs(
    percentiles_text: str, metric: Type[Metric]
) -> list[AggregationFunc]:
    percentiles = str_to_percentiles(percentiles_text)
    return [
        AggregationFunc(
            lambda x, q=p / 100: x.quantile(q),  # type:ignore[misc]
            f"p{p}",
            p,
            metric.field_name,
        )
        for p in percentiles
    ]
