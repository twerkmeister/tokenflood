from __future__ import annotations

import re
from typing import List

from tokenflood.analysis import AggregationFunc, calculate_percentile

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

def percentiles_to_aggregation_funcs(percentiles_text: str) -> list[AggregationFunc]:
    percentiles = str_to_percentiles(percentiles_text)
    return [calculate_percentile(p) for p in percentiles]