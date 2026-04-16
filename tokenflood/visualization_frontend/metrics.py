from dataclasses import dataclass

from tokenflood.constants import LLM_REQUESTS_FILE, NETWORK_LATENCY_FILE
from tokenflood.models.llm_request_data import LLMRequestData
from tokenflood.models.ping_request_data import PingData


@dataclass
class Metric:
    field_name: str
    file: str
    name: str


class RequestLatency(Metric):
    field_name = LLMRequestData.F.latency
    file = LLM_REQUESTS_FILE
    name = "Request Latency"


class NetworkLatency(Metric):
    field_name = PingData.F.latency
    file = NETWORK_LATENCY_FILE
    name = "Network Latency"


metric_mapping = {
    RequestLatency.name: RequestLatency,
    NetworkLatency.name: NetworkLatency,
}
