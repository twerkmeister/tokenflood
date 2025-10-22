from typing import Self, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from tokenflood.models.validation_types import NonNegativeIntegers
from tokenflood.util import calculate_mean_absolute_error


class Results(BaseModel, frozen=True):
    prompts: Sequence[str]
    generated_texts: Sequence[str]
    latencies: NonNegativeIntegers
    expected_input_lengths: NonNegativeIntegers
    expected_prefix_lengths: NonNegativeIntegers
    expected_output_lengths: NonNegativeIntegers
    measured_input_lengths: NonNegativeIntegers
    measured_prefix_lengths: NonNegativeIntegers
    measured_output_lengths: NonNegativeIntegers

    @model_validator(mode="after")
    def check_all_sequences_same_size(self) -> Self:
        data_sequences = self.model_dump().values()
        data_sequences_lengths = [len(ds) for ds in data_sequences]
        if any([dsl != data_sequences_lengths[0] for dsl in data_sequences_lengths]):
            raise ValueError(
                "All data sequences of the result need to be the same size."
            )
        return self

    def get_input_length_error(self) -> float:
        return calculate_mean_absolute_error(
            self.expected_input_lengths, self.measured_input_lengths
        )

    def get_prefix_length_error(self) -> float:
        return calculate_mean_absolute_error(
            self.expected_prefix_lengths, self.measured_prefix_lengths
        )

    def get_output_length_error(self) -> float:
        return calculate_mean_absolute_error(
            self.expected_output_lengths, self.measured_output_lengths
        )

    def get_latency_percentile(self, percentile: int) -> float:
        return float(np.percentile(self.latencies, percentile))

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.model_dump())
        return df[df.columns[::-1]]
