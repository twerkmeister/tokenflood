from dataclasses import dataclass

from tokenflood.constants import LLM_REQUESTS_FILE, NETWORK_LATENCY_FILE
from tokenflood.models.data.llm_request_data import LLMRequestData
from tokenflood.models.data.ping_request_data import PingData


@dataclass(frozen=True)
class Metric:
    field_name: str
    file: str
    name: str
    explanation: str


class RequestLatency(Metric):
    field_name = LLMRequestData.F.latency
    file = LLM_REQUESTS_FILE
    name = "Request Latency"
    explanation = "Total latency including network, prefilling and decoding latency."


class TimeToFirstToken(Metric):
    field_name = LLMRequestData.F.time_to_first_token
    file = LLM_REQUESTS_FILE
    name = "Time to first token"
    explanation = "Time to first token including network latency."


class DecodingLatency(Metric):
    field_name = LLMRequestData.F.decoding_latency
    file = LLM_REQUESTS_FILE
    name = "Decoding latency"
    explanation = "Latency of the output generation past the first token."


class AverageTimePerOutputToken(Metric):
    field_name = LLMRequestData.F.average_time_per_output_token
    file = LLM_REQUESTS_FILE
    name = "Average time per output token"
    explanation = "Decoding latency divided by number of tokens generated."


class NetworkLatency(Metric):
    field_name = PingData.F.latency
    file = NETWORK_LATENCY_FILE
    name = "Network Latency"
    explanation = (
        "Network latency measured by pinging the endpoint with non-generating requests."
    )


metric_mapping = {
    RequestLatency.name: RequestLatency,
    TimeToFirstToken.name: TimeToFirstToken,
    AverageTimePerOutputToken.name: AverageTimePerOutputToken,
    DecodingLatency.name: DecodingLatency,
    NetworkLatency.name: NetworkLatency,
}
