from typing import List

import numpy as np

from tokenflood.models.run_specs.load_spec import LoadPhase

highest_burstiness_control = 21


def create_even_schedule(num_requests: int, within_seconds: float) -> List[float]:
    if num_requests < 1:
        return []
    return list(np.diff(np.linspace(0, within_seconds, num_requests))) + [0.0]


def burstiness_to_burstiness_control(burstiness: int) -> int:
    """Returns the burstiness factor 0-10 as the inverted burstiness control."""
    return highest_burstiness_control - burstiness * 2


def create_load_test_phase_schedule(
    load_phase: LoadPhase, burstiness: int
) -> List[float]:
    """Create a bursty randomized schedule with a guaranteed total length."""
    burstiness_control = burstiness_to_burstiness_control(burstiness)
    # if burstiness control is highest, create an even schedule instead
    if burstiness_control >= highest_burstiness_control:
        return create_even_schedule(
            load_phase.total_num_requests, load_phase.duration_seconds
        )
    if load_phase.total_num_requests == 1:
        return [0.0]
    pauses = np.random.gamma(
        shape=burstiness_control,
        scale=(1 / load_phase.requests_per_second) / burstiness_control,
        size=load_phase.total_num_requests - 1,
    )
    total_length = pauses.sum()
    pauses = pauses / (total_length / load_phase.duration_seconds)
    return list(pauses) + [0.0]
