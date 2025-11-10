import datetime
from typing import Callable, Optional, Sequence, TypeVar
import numpy as np

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.util import numeric


def get_exact_date_str() -> str:
    # trim microseconds
    datetime_with_milliseconds = datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%d_%H-%M-%S.%f"
    )[:-3]
    return f"{datetime_with_milliseconds}(UTC)"


def get_date_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_run_name(date_str: str, endpoint_spec: EndpointSpec):
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
        float(
            calculate_mean_error(observations, targets)
            / (float(np.average(targets)) + 1e-3)
        ),
        2,
    )


T = TypeVar("T")
X = TypeVar("X")


def find_idx(s: Sequence[T], predicate: Callable[[T], bool]) -> Optional[int]:
    for i in range(len(s)):
        if predicate(s[i]):
            return i
    return None
