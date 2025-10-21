import argparse
import asyncio

from tokenflood.graphing import visualize_percentiles_across_request_rates
from tokenflood.io import read_endpoint_spec, read_run_suite
from tokenflood.runner import run_suite


def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog="tokenflood",
        description="Tokenflood helps you load test instruction-tuned LLMs.",
    )

    subparsers = parser.add_subparsers()
    run_cmd_parser = subparsers.add_parser("run", help="Execute a run suite and graph results")
    run_cmd_parser.add_argument("run_suite", type=str)
    run_cmd_parser.add_argument("endpoint_spec", type=str)
    run_cmd_parser.add_argument("output_file", type=str)
    run_cmd_parser.set_defaults(func=run_and_graph_suite)

    return parser

def run_and_graph_suite(args: argparse.Namespace):
    endpoint_spec = read_endpoint_spec(args.endpoint_spec)
    suite = read_run_suite(args.run_suite)
    run_suite_data = asyncio.run(run_suite(endpoint_spec, suite))
    visualize_percentiles_across_request_rates(suite, run_suite_data, args.output_file)

def main():
    arg_parser = create_argument_parser()
    args = arg_parser.parse_args()
    args.func(args)
