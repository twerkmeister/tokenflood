import math
from typing import Callable, Self, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from tokenflood.models.validation_types import NonNegativeIntegersOrEmpty
from tokenflood.util import calculate_mean_error, calculate_relative_error


class Results(BaseModel, frozen=True):
    latencies: NonNegativeIntegersOrEmpty
    expected_input_lengths: NonNegativeIntegersOrEmpty
    measured_input_lengths: NonNegativeIntegersOrEmpty
    expected_prefix_lengths: NonNegativeIntegersOrEmpty
    measured_prefix_lengths: NonNegativeIntegersOrEmpty
    expected_output_lengths: NonNegativeIntegersOrEmpty
    measured_output_lengths: NonNegativeIntegersOrEmpty
    generated_texts: Sequence[str]
    prompts: Sequence[str]

    @model_validator(mode="after")
    def check_all_sequences_same_size(self) -> Self:
        data_sequences = self.model_dump().values()
        data_sequences_lengths = [len(ds) for ds in data_sequences]
        if any([dsl != data_sequences_lengths[0] for dsl in data_sequences_lengths]):
            raise ValueError(
                "All data sequences of the result need to be the same size."
            )
        return self

    @property
    def is_empty(self):
        return len(self.latencies) == 0

    @staticmethod
    def nan_if_empty(func: Callable[..., float]) -> Callable[..., float]:
        def wrapped(myself: "Results", *args):
            if myself.is_empty:
                return math.nan
            else:
                return func(myself, *args)

        return wrapped

    @nan_if_empty
    def get_input_length_error(self) -> float:
        return calculate_mean_error(
            self.expected_input_lengths, self.measured_input_lengths
        )

    @nan_if_empty
    def get_relative_input_length_error(self) -> float:
        return calculate_relative_error(
            self.measured_input_lengths, self.expected_input_lengths
        )

    @nan_if_empty
    def get_prefix_length_error(self) -> float:
        return calculate_mean_error(
            self.expected_prefix_lengths, self.measured_prefix_lengths
        )

    @nan_if_empty
    def get_relative_prefix_length_error(self) -> float:
        return calculate_relative_error(
            self.measured_prefix_lengths, self.expected_prefix_lengths
        )

    @nan_if_empty
    def get_output_length_error(self) -> float:
        return calculate_mean_error(
            self.expected_output_lengths, self.measured_output_lengths
        )

    @nan_if_empty
    def get_relative_output_length_error(self) -> float:
        return calculate_relative_error(
            self.measured_output_lengths, self.expected_output_lengths
        )

    @nan_if_empty
    def get_latency_percentile(self, percentile: int) -> float:
        return round(float(np.percentile(self.latencies, percentile)), 2)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump())
