from typing import Self, Sequence

from pydantic import BaseModel, model_validator

from tokenflood.models.validation_types import StrictlyPositiveIntegers


class Results(BaseModel, frozen=True):
    prompts: Sequence[str]
    generated_texts: Sequence[str]
    latencies: StrictlyPositiveIntegers
    expected_input_lengths: StrictlyPositiveIntegers
    expected_prefix_lengths: StrictlyPositiveIntegers
    expected_output_lengths: StrictlyPositiveIntegers
    measured_input_lengths: StrictlyPositiveIntegers
    measured_prefix_lengths: StrictlyPositiveIntegers
    measured_output_lengths: StrictlyPositiveIntegers

    @model_validator(mode="after")
    def check_all_sequences_same_size(self) -> Self:
        data_sequences = self.model_dump().values()
        data_sequences_lengths = [len(ds) for ds in data_sequences]
        if any([dsl != data_sequences_lengths[0] for dsl in data_sequences_lengths]):
            raise ValueError(
                "All data sequences of the result need to be the same size."
            )
        return self
