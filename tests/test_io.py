import os


from tokenflood.io import read_endpoint_spec, read_run_suite


def test_read_endpoint_spec(endpoint_specs_folder):
    filename = os.path.join(endpoint_specs_folder, "base.yml")
    endpoint_spec = read_endpoint_spec(filename)
    assert endpoint_spec.model == "openai/HuggingFaceTB/SmolLM-135M-Instruct"
    assert endpoint_spec.base_url == "http://127.0.0.1:8000/v1"


def test_read_run_suite(run_suites_folder):
    filename = os.path.join(run_suites_folder, "base.yml")
    run_suite = read_run_suite(filename)
    assert len(run_suite.load_types) == 2
    assert run_suite.name == "ABC"
    assert run_suite.requests_per_second_rates == tuple(range(1, 5))
    assert run_suite.test_length_in_seconds == 30
    assert run_suite.percentiles == (50, 90, 98)
