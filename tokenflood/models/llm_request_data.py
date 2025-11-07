from typing import Self

from litellm.types.utils import ModelResponse
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from tokenflood.models.validation_types import NonEmptyString


class LLMRequestResult(BaseModel, frozen=True):
    latency: NonNegativeInt
    measured_input_tokens: NonNegativeInt
    measured_prefix_tokens: NonNegativeInt
    measured_output_tokens: NonNegativeInt
    generated_text: str

    @classmethod
    def from_model_response(cls, model_response: ModelResponse) -> Self:
        usage = model_response.usage  # type:ignore[attr-defined]
        return cls(
            latency=int(model_response._response_ms),  # type:ignore[attr-defined]
            measured_input_tokens=usage.prompt_tokens,
            measured_prefix_tokens=usage.prompt_tokens_details.cached_tokens or 0
            if usage.prompt_tokens_details
            else 0,
            measured_output_tokens=usage.completion_tokens,
            generated_text=model_response.choices[0]["message"]["content"],
        )


class LLMRequestContext(BaseModel, frozen=True):
    datetime: NonEmptyString
    expected_input_tokens: NonNegativeInt
    expected_prefix_tokens: NonNegativeInt
    expected_output_tokens: NonNegativeInt
    requests_per_second_phase: NonNegativeFloat
    request_number: NonNegativeInt
    model: NonEmptyString
    prompt: str


class LLMRequestData(BaseModel, frozen=True):
    """Just a class to ensure data ordering in the results."""

    datetime: NonEmptyString
    requests_per_second_phase: NonNegativeFloat
    request_number: NonNegativeInt
    model: NonEmptyString
    latency: NonNegativeInt
    expected_input_tokens: NonNegativeInt
    measured_input_tokens: NonNegativeInt
    expected_prefix_tokens: NonNegativeInt
    measured_prefix_tokens: NonNegativeInt
    expected_output_tokens: NonNegativeInt
    measured_output_tokens: NonNegativeInt
    generated_text: str
    prompt: str

    @classmethod
    def from_result_and_context(
        cls, result: LLMRequestResult, context: LLMRequestContext
    ) -> Self:
        return cls(**{**result.model_dump(), **context.model_dump()})

    @classmethod
    def from_response_and_context(
        cls, response: ModelResponse, context: LLMRequestContext
    ) -> Self:
        return cls.from_result_and_context(
            LLMRequestResult.from_model_response(response), context
        )
