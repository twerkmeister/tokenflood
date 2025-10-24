import argparse
import asyncio
import os
import sys
from io import StringIO
from typing import List
from rich import print
import logging
from rich.logging import RichHandler
from tokenflood import __version__

from dotenv import load_dotenv

from tokenflood.constants import (
    ENDPOINT_SPEC_FILE,
    ERROR_FILE,
    LATENCY_GRAPH_FILE,
    MAX_INPUT_TOKENS_DEFAULT,
    MAX_INPUT_TOKENS_ENV_VAR,
    MAX_OUTPUT_TOKENS_DEFAULT,
    MAX_OUTPUT_TOKENS_ENV_VAR,
    RUN_DATA_FILE,
    RUN_SUITE_FILE,
    SUMMARY_FILE,
)
from tokenflood.graphing import (
    visualize_percentiles_across_request_rates,
    write_out_error,
    write_out_raw_data_points,
    write_out_summary,
)
from tokenflood.io import (
    get_first_available_filename_like,
    make_run_folder,
    read_endpoint_spec,
    read_run_suite,
    write_pydantic_yaml,
)
from tokenflood.runner import check_token_usage_upfront, run_suite
from tokenflood.starter_pack import (
    starter_endpoint_spec_filename,
    starter_endpoint_spec_vllm,
    starter_model_id,
    starter_run_suite,
    starter_run_suite_filename,
)
from tokenflood.util import get_run_name

log = logging.getLogger(__name__)


the_wave = f"""[blue]
⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠇⡅⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀           
⠧⡇⠀⠀⠒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⡤⡆⠦⠆⢀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  
⠧⣷⣆⠅⢦⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠈⠀⠀⠀⠀⠀⢤⣤⣆⢇⣶⣤⡤⡯⣦⣌⡡⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠷⣿⣷⣆⣐⡆⠀⠀⠀⠀⢀⠤⠊⠀⠀⢀⣠⣾⢯⣦⣴⣜⣺⣾⣿⣤⠟⠋⣷⢛⡣⠭⠢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠯⣿⣷⢫⡯⠄⠀⠀⢀⠐⠁⠀⠀⠀⠠⣤⣿⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣙⣷⡗⢤⡤⠀⠈⣰⠶⡤⠀⠀⠀⠀⠀⠀⠀[/][yellow]tokenflood v{__version__}[/][blue]⠀⠀⠀⠀⠀⠀⠀⠀
⣩⣿⡏⠉⠉⠀⢠⡔⠁⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠑⣏⠶⡉⠖⣡⠂⣈⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣮⣿⣧⣤⣤⠖⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⢉⡻⣿⣿⣿⣿⣿⣿⣿⣿⠟⠓⠈⠅⠈⠀⠀⠘⢒⣽⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⡿⠛⠉⠀⠀⠀⣀⠔⢀⡴⣃⠀⠀⢀⠷⠲⡄⠸⠟⢋⣿⣿⣿⣿⣿⡇⠀⠀⠀⠐⠁⠀⠀⠂⠀⠀⠰⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⡆⣷⣆⡐⠶⠤⢤⣷⣀⣀⣩⢐⣟⣥⠜⣤⣀⣠⣤⠀⠈⠉⢀⣹⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢃⣿⣞⣫⡔⢆⡸⡿⣿⣿⣄⣰⣿⠁⢀⣛⠿⣻⣿⣿⣧⣬⣿⣿⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⢀
⢼⣿⣟⢿⣧⣾⣵⣷⣿⣿⣟⡿⢿⣶⣞⣍⡴⢿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⣠⠈⠀⢀⣀⣼
⠋⣿⣟⡛⢿⣿⣿⣿⣿⣿⣭⣿⣿⣿⣿⣯⣽⣿⣿⣿⣿⠟⠛⠿⢽⣿⣿⣆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⣀⢀⡠⣤⣤⣰⣿⠟⠁⠀⠀⡼⢾⣿
⣻⣿⣟⣇⠈⣉⣯⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠃⠀⠀⠀⠀⠀⠻⣿⣿⣿⣿⣴⣶⣤⣤⣤⣤⣴⣴⣴⣶⣦⣦⣤⣦⣀⣦⣤⣶⣿⣿⣿⣿⣿⣿⣿⠿⠁⠀⠀⡀⣤⣬⣾⣿
⡝⣿⣿⣇⣤⣶⣿⣷⣾⣭⡿⠻⢿⣿⣿⣿⣿⠿⠃⠀⠀⠀⠀⡄⠀⠀⠀⢊⡻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠋⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⢿⠟⢉⠀⡀⢤⣴⣿⣿⣿⠿⠻
⡁⣻⣿⣿⣿⣿⣷⣿⣿⣿⣿⠾⣿⡿⠞⠁⠀⠀⠀⠀⠀⠔⠫⡅⠀⠀⠀⠀⠁⣀⠀⠈⠻⣿⣿⣿⣿⣻⢟⣁⣄⡄⣀⠙⠻⣿⣿⡿⠿⠛⡋⠕⠂⢀⣀⣄⣓⣳⢿⠟⢛⣩⠴⠈⠀
⠂⡁⠈⠛⠛⠛⠛⠋⠁⠀⠈⠈⡀⠀⠀⠀⠀⢀⠘⠀⠀⠀⠆⠀⡀⡢⣀⣆⠄⠈⠨⢦⡀⣈⠙⠛⠿⢿⣿⣿⣿⣿⣿⡿⡿⠿⠟⠆⠒⠁⠀⢶⣾⠿⠟⠛⢉⣀⣠⡶⠚⠁⠀⠀⣠
⠀⡇⡄⣀⡀⠀⠀⠀⠀⠀⠀⠀⢬⠠⠀⡀⠀⠋⠁⠀⡀⠀⠀⡀⠆⢱⣿⣿⣧⣧⣄⠛⣿⣞⣵⣤⣷⣄⠀⠀⠀⠐⠀⠀⠀⠀⠀⠈⠉⠁⠁⠀⠠⢤⣶⣾⣿⡿⠋⢀⣀⣰⣶⣾⣿
⡀⡆⠀⡉⡁⢿⣉⢀⠀⣰⣷⣿⣟⠠⡽⢂⡀⡄⠀⠰⣖⢱⢖⢂⡆⠈⣿⣿⣿⣿⣿⣶⣄⡙⠻⢿⣿⣿⣷⣦⣀⠀⠠⣤⣀⡀⢈⣓⣶⣶⣿⣿⣿⣿⣿⠟⠉⠀⠀⠀⣉⣭⣽⣿⣿
⡇⣯⣿⣿⣿⣾⣿⣿⣿⠿⠟⡡⢞⣹⠾⢻⣚⣛⢺⠞⢋⣭⣾⣧⡃⢄⡈⢿⣿⣿⣿⣿⣿⣿⣯⣿⣮⣽⣿⣿⣿⣿⣷⣬⣽⣿⣿⣿⣽⡿⣿⡿⠟⠋⢀⣀⣐⣺⣿⣿⣟⣫⣭⣿⣿
⢳⣿⣿⣿⣿⣿⣿⣿⣿⣤⣿⣿⣿⣿⣿⣦⠒⠉⢁⡀⠀⣙⣛⢿⣷⣶⣅⠀⠙⠻⣿⣿⣿⣿⣟⡚⠛⠻⠞⠿⠿⡿⡿⠯⠁⠟⣊⠾⠝⢋⣁⣀⣤⣤⣿⣿⣿⡿⠿⠿⠻⠛⠻⠻⠿
⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣐⣾⡿⡟⢶⠾⢋⢹⠿⢿⣿⣿⣷⣦⡈⠙⠛⠿⠿⢿⣶⣶⣶⣶⣶⢶⠟⠚⠀⠁⠀⠀⠙⠛⠛⠛⠛⠛⠋⠉⠁⠀⠀⠀⠀⠀⢀⠀⠀[/]"""


def configure_logging():
    tokenflood_logger = logging.getLogger("tokenflood")
    tokenflood_logger.setLevel(logging.INFO)
    tokenflood_logger.addHandler(RichHandler(markup=True))


def create_argument_parser():
    parser = argparse.ArgumentParser(
        prog="tokenflood",
        description="[blue]A load testing tool for LLMs that simulates arbitrary work loads.[/]",
    )
    parser.set_defaults(func=print_help_of(parser))
    subparsers = parser.add_subparsers()

    # RUN
    run_cmd_parser = subparsers.add_parser(
        "run", help="[blue]Execute a run suite and graph results[/]"
    )
    run_cmd_parser.add_argument("run_suite", type=str)
    run_cmd_parser.add_argument("endpoint", type=str)
    run_cmd_parser.add_argument(
        "-y",
        "--autoaccept",
        help="Auto accept run start if tokens are within configured limits.",
        action="store_true",
    )
    run_cmd_parser.set_defaults(func=run_and_graph_suite)

    # Ripple
    ripple_cmd_parser = subparsers.add_parser(
        "ripple",
        help="[blue]Create starter files for run suite and endpoint specifications.[/]",
    )
    ripple_cmd_parser.set_defaults(func=create_starter_files)

    return parser


def parse_args(args: List[str]) -> argparse.Namespace:
    arg_parser = create_argument_parser()
    return arg_parser.parse_args(args)


def print_help_of(arg_parser: argparse.ArgumentParser):
    """Convenience method so we can always pass the args into the target func."""

    def print_help(args: argparse.Namespace):
        s = StringIO()
        arg_parser.print_help(s)
        s.seek(0)
        print(s.read())

    return print_help


def create_starter_files(args: argparse.Namespace):
    available_endpoint_spec_filename = get_first_available_filename_like(
        starter_endpoint_spec_filename
    )
    available_run_suite_filename = get_first_available_filename_like(
        starter_run_suite_filename
    )
    log.info(
        "Creating starter files for run suite and endpoint specifications: \n"
        f"[green]* {available_run_suite_filename} [/]\n"
        f"[green]* {available_endpoint_spec_filename} [/]"
    )

    write_pydantic_yaml(available_endpoint_spec_filename, starter_endpoint_spec_vllm)
    write_pydantic_yaml(available_run_suite_filename, starter_run_suite)

    log.info(
        "Inspect those files, boot up a vllm server with a tiny model using \n"
        f"[blue]vllm serve {starter_model_id}[/]\n"
        "and run a first load test against it using\n"
        f"[blue]tokenflood run {available_run_suite_filename} {available_endpoint_spec_filename}[/]"
    )


def run_and_graph_suite(args: argparse.Namespace):
    endpoint_spec = read_endpoint_spec(args.endpoint)
    suite = read_run_suite(args.run_suite)
    run_name = get_run_name(endpoint_spec)

    accepted_token_usage = check_token_usage_upfront(
        suite,
        int(os.getenv(MAX_INPUT_TOKENS_ENV_VAR, MAX_INPUT_TOKENS_DEFAULT)),
        int(os.getenv(MAX_OUTPUT_TOKENS_ENV_VAR, MAX_OUTPUT_TOKENS_DEFAULT)),
        args.autoaccept,
    )
    if not accepted_token_usage:
        log.info("Stopping because token usage was not accepted.")
        return

    run_folder = make_run_folder(run_name)
    latency_graph_file = os.path.join(run_folder, LATENCY_GRAPH_FILE)
    run_data_file = os.path.join(run_folder, RUN_DATA_FILE)
    endpoint_spec_file = os.path.join(run_folder, ENDPOINT_SPEC_FILE)
    run_suite_file = os.path.join(run_folder, RUN_SUITE_FILE)
    error_file = os.path.join(run_folder, ERROR_FILE)
    summary_file = os.path.join(run_folder, SUMMARY_FILE)
    run_suite_data = asyncio.run(run_suite(endpoint_spec, suite))

    # write out input configs and results to run folder
    write_out_error(run_suite_data, error_file)
    write_pydantic_yaml(endpoint_spec_file, endpoint_spec)
    write_pydantic_yaml(run_suite_file, suite)
    write_out_raw_data_points(run_suite_data, run_data_file)
    write_out_summary(suite, endpoint_spec, run_suite_data, summary_file)
    visualize_percentiles_across_request_rates(
        suite, run_suite_data, latency_graph_file
    )


def main():
    load_dotenv(".env")
    configure_logging()
    args = parse_args(sys.argv[1:])
    print(the_wave)
    args.func(args)
