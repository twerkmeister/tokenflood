from dataclasses import dataclass

from tokenflood.constants import LLM_REQUESTS_FILE, NETWORK_LATENCY_FILE
from tokenflood.models.llm_request_data import LLMRequestData
from tokenflood.models.ping_request_data import PingData


@dataclass
class Metric:
    field_name: str
    file: str

class RequestLatency(Metric):
    field_name = LLMRequestData.F.latency
    file = LLM_REQUESTS_FILE

class NetworkLatency(Metric):
    field_name = PingData.F.latency
    file = NETWORK_LATENCY_FILE

metric_mapping = {
    RequestLatency.__name__: RequestLatency,
    NetworkLatency.__name__: NetworkLatency
}