import logging
from typing import Self

from litellm.types.utils import ModelResponse
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from tokenflood.constants import WARNING_LIMIT
from tokenflood.logging import WARN_ONCE_KEY
from tokenflood.models.validation_types import NonEmptyString
from tokenflood.util import calculate_relative_error

log = logging.getLogger(__name__)


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

    def warn_on_diverging_measurements(self):
        relative_input_token_error = calculate_relative_error(
            [self.measured_input_tokens], [self.expected_input_tokens]
        )
        if abs(relative_input_token_error) > WARNING_LIMIT:
            longer_or_shorter = (
                "longer" if relative_input_token_error > 0 else "shorter"
            )
            log.warning(
                f"Observed input tokens that are {abs(int(relative_input_token_error * 100))}% {longer_or_shorter} than what was expected. The measured latencies might not be representative. This warning type will only appear once per phase.",
                extra={WARN_ONCE_KEY: "input_tokens_off"},
            )

        relative_output_token_error = calculate_relative_error(
            [self.measured_output_tokens], [self.expected_output_tokens]
        )
        if abs(relative_output_token_error) > WARNING_LIMIT:
            longer_or_shorter = (
                "longer" if relative_output_token_error > 0 else "shorter"
            )
            log.warning(
                f"Observed output tokens that are {abs(int(relative_output_token_error * 100))}% {longer_or_shorter} than what was expected. The measured latencies might not be representative. This warning type will only appear once per phase.",
                extra={WARN_ONCE_KEY: "output_tokens_off"},
            )
