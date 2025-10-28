from datetime import datetime
from typing import Callable, Optional, Sequence, TypeVar

import numpy as np

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.util import numeric


def get_run_name(endpoint_spec: EndpointSpec):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{date_str}_{endpoint_spec.provider_model_str_as_folder_name}"


def calculate_mean_error(
    observations: Sequence[numeric], targets: Sequence[numeric]
) -> float:
    if len(observations) != len(targets):
        raise ValueError(
            f"Sequences must be same size to calculate mean error,"
            f"but have lengths {len(observations)} and {len(targets)} respectively."
        )
    return float(np.average(np.asarray(observations) - np.asarray(targets)))


def calculate_relative_error(
    observations: Sequence[numeric], targets: Sequence[numeric]
) -> float:
    return round(
        float(calculate_mean_error(observations, targets) / float(np.average(targets))),
        2,
    )


T = TypeVar("T")


def find_idx(s: Sequence[T], predicate: Callable[[T], bool]) -> Optional[int]:
    for i in range(len(s)):
        if predicate(s[i]):
            return i
    return None
