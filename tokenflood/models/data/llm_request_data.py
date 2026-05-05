import logging
from typing import Self

from litellm.types.utils import ModelResponse
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from tokenflood.constants import WARNING_LIMIT
from tokenflood.logging_utils import WARN_ONCE_KEY
from tokenflood.models.validation_types import NonEmptyString
from tokenflood.util import calculate_relative_error

log = logging.getLogger(__name__)


class LLMRequestResult(BaseModel, frozen=True):
    latency: NonNegativeInt
    time_to_first_token: NonNegativeInt
    decoding_latency: NonNegativeInt
    average_time_per_output_token: NonNegativeFloat
    measured_input_tokens: NonNegativeInt
    measured_prefix_tokens: NonNegativeInt
    measured_output_tokens: NonNegativeInt
    measured_reasoning_tokens: NonNegativeInt
    generated_text: str
    generated_reasoning: str

    @classmethod
    def from_model_response(cls, model_response: ModelResponse) -> Self:
        usage = model_response.usage  # type:ignore[attr-defined]
        hp = model_response._hidden_params
        msg = model_response.choices[0]["message"]
        return cls(
            latency=int(hp[LLMRequestData.F.latency]),
            time_to_first_token=int(hp[LLMRequestData.F.time_to_first_token]),
            decoding_latency=int(hp[LLMRequestData.F.decoding_latency]),
            average_time_per_output_token=hp[
                LLMRequestData.F.average_time_per_output_token
            ],
            measured_input_tokens=usage.prompt_tokens,
            measured_prefix_tokens=(usage.prompt_tokens_details.cached_tokens or 0)
            if usage.prompt_tokens_details
            else 0,
            measured_output_tokens=usage.completion_tokens,
            measured_reasoning_tokens=(
                usage.completion_tokens_details.reasoning_tokens or 0
            )
            if usage.completion_tokens_details
            else 0,
            generated_text=msg.get("content", ""),
            generated_reasoning=msg.get("reasoning_content", ""),
        )


class LLMRequestContext(BaseModel, frozen=True):
    datetime: NonEmptyString
    expected_input_tokens: NonNegativeInt
    expected_prefix_tokens: NonNegativeInt
    expected_output_tokens: NonNegativeInt
    requests_per_second_phase: NonNegativeFloat
    request_number: NonNegativeInt
    model: NonEmptyString
    group_id: NonEmptyString
    prompt: str


class LLMRequestData(BaseModel, frozen=True):
    datetime: NonEmptyString
    requests_per_second_phase: NonNegativeFloat
    request_number: NonNegativeInt
    model: NonEmptyString
    latency: NonNegativeInt
    time_to_first_token: NonNegativeInt
    decoding_latency: NonNegativeInt
    average_time_per_output_token: NonNegativeFloat
    expected_input_tokens: NonNegativeInt
    measured_input_tokens: NonNegativeInt
    expected_prefix_tokens: NonNegativeInt
    measured_prefix_tokens: NonNegativeInt
    expected_output_tokens: NonNegativeInt
    measured_output_tokens: NonNegativeInt
    measured_reasoning_tokens: NonNegativeInt
    group_id: NonEmptyString
    generated_text: str
    generated_reasoning: str
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
                f"Observed input tokens that are {abs(relative_input_token_error * 100):.2f}% {longer_or_shorter} than what was expected. The measured latencies might not be representative. This warning type will only appear once per phase.",
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
                f"Observed output tokens that are {abs(relative_output_token_error * 100):.2f}% {longer_or_shorter} than what was expected. The measured latencies might not be representative. This warning type will only appear once per phase.",
                extra={WARN_ONCE_KEY: "output_tokens_off"},
            )

    # field names for access in analytics code
    class F:
        latency = "latency"
        time_to_first_token = "time_to_first_token"
        decoding_latency = "decoding_latency"
        average_time_per_output_token = "average_time_per_output_token"
