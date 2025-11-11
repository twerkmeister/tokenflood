from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.load_type import LoadType
from tokenflood.models.run_suite import HeuristicRunSuite

starter_run_suite_filename = "run_suite.yml"
starter_run_suite = HeuristicRunSuite(
    name="ripple",
    requests_per_second_rates=(1, 2),
    test_length_in_seconds=30,
    load_types=(
        LoadType(prompt_length=512, prefix_length=128, output_length=32, weight=1),
        LoadType(prompt_length=640, prefix_length=568, output_length=12, weight=1),
    ),
    percentiles=(50, 90, 99),
)

starter_endpoint_spec_filename = "endpoint.yml"
starter_model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
starter_endpoint_spec_vllm = EndpointSpec(
    provider="hosted_vllm", model=starter_model_id, base_url="http://127.0.0.1:8000/v1"
)
