from datetime import datetime
from typing import Sequence

import numpy as np

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.util import numeric


def get_run_name(endpoint_spec: EndpointSpec):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{date_str}_{endpoint_spec.provider_model_str_as_folder_name}"


def calculate_mean_absolute_error(
    s1: Sequence[numeric], s2: Sequence[numeric]
) -> float:
    if len(s1) != len(s2):
        raise ValueError(
            f"Sequences must be same size to calculate mean absolute error,"
            f"but have lengths {len(s1)} and {len(s2)} respectively."
        )
    return float(np.average(np.abs(np.asarray(s1) - np.asarray(s2))))
