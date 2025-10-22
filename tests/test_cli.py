from tokenflood.cli import parse_args


def test_parse_args_run():
    endpoint_spec = "endpoint.yml"
    run_suite = "suite.yml"
    args = parse_args(["run", run_suite, endpoint_spec])
    assert args.endpoint_spec == endpoint_spec
    assert args.run_suite == run_suite
    assert args.func.__name__ == "run_and_graph_suite"


def test_parse_args_empty():
    args = parse_args([])
    assert args.func.__name__ == "print_help"


def test_run_and_graph_suite(monkeypatch, unique_temporary_folder):
    monkeypatch.chdir(unique_temporary_folder)
