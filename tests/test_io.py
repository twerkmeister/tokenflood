def test_read_base_endpoint_spec(base_endpoint_spec):
    assert base_endpoint_spec.model == "hosted_vllm/HuggingFaceTB/SmolLM-135M-Instruct"
    assert base_endpoint_spec.base_url == "http://127.0.0.1:8000/v1"


def test_read_base_run_suite(base_run_suite):
    assert len(base_run_suite.load_types) == 2
    assert base_run_suite.name == "ABC"
    assert base_run_suite.requests_per_second_rates == tuple(range(1, 5))
    assert base_run_suite.test_length_in_seconds == 30
    assert base_run_suite.percentiles == (50, 90, 98)
