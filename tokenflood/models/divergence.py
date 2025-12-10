from functools import cached_property
from typing import Callable

import pandas as pd
from pydantic import BaseModel

from tokenflood.analysis import mean_float
from tokenflood.models.llm_request_data import LLMRequestData
from tokenflood.models.util import numeric
from tokenflood.util import calculate_relative_error


class TokenDivergence(BaseModel, arbitrary_types_allowed=True, frozen=True):
    llm_request_data: pd.DataFrame

    def safe_stat(
        self, column_name: str, operation: Callable[[pd.Series], numeric]
    ) -> numeric:
        if column_name not in LLMRequestData.model_fields:
            raise ValueError(f"Column {column_name} is not part of llm request data.")
        return round(operation(self.llm_request_data[column_name]), 2)

    @cached_property
    def mean_expected_input_tokens(self) -> float:
        return self.safe_stat("expected_input_tokens", mean_float)

    @cached_property
    def mean_measured_input_tokens(self) -> float:
        return self.safe_stat("measured_input_tokens", mean_float)

    @cached_property
    def mean_expected_output_tokens(self) -> float:
        return self.safe_stat("expected_output_tokens", mean_float)

    @cached_property
    def mean_measured_output_tokens(self) -> float:
        return self.safe_stat("measured_output_tokens", mean_float)

    @cached_property
    def mean_expected_prefix_tokens(self) -> float:
        return self.safe_stat("expected_prefix_tokens", mean_float)

    @cached_property
    def mean_measured_prefix_tokens(self) -> float:
        return self.safe_stat("measured_prefix_tokens", mean_float)

    @staticmethod
    def nice_relative_error(observation: float, target: float) -> float:
        return round(100 * calculate_relative_error([observation], [target]), 2)

    @cached_property
    def relative_input_token_error(self) -> float:
        return self.nice_relative_error(
            self.mean_measured_input_tokens, self.mean_expected_input_tokens
        )

    @cached_property
    def relative_output_token_error(self) -> float:
        return self.nice_relative_error(
            self.mean_measured_output_tokens, self.mean_expected_output_tokens
        )

    @cached_property
    def relative_prefix_token_error(self) -> float:
        return self.nice_relative_error(
            self.mean_measured_prefix_tokens, self.mean_expected_prefix_tokens
        )
