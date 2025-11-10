import argparse
import asyncio
import os
import sys
from io import StringIO
from typing import List

import pandas as pd
from rich import print
import logging

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from tokenflood import __version__

from dotenv import load_dotenv

from tokenflood.constants import (
    ENDPOINT_SPEC_FILE,
    ERROR_FILE,
    LATENCY_GRAPH_FILE,
    NETWORK_LATENCY_FILE,
    LLM_REQUESTS_FILE,
    RUN_SUITE_FILE,
    SUMMARY_FILE,
    WARNING_LIMIT,
)
from tokenflood.analysis import (
    create_summary,
    make_super_title,
    visualize_percentiles_across_request_rates,
)
from tokenflood.io import (
    FileIOContext,
    get_first_available_filename_like,
    make_run_folder,
    read_endpoint_spec,
    read_run_suite,
    write_pydantic_yaml,
)
from tokenflood.logging import global_warn_once_filter
from tokenflood.models.run_summary import RunSummary
from tokenflood.runner import check_token_usage_upfront, run_suite
from tokenflood.starter_pack import (
    starter_endpoint_spec_filename,
    starter_endpoint_spec_vllm,
    starter_model_id,
    starter_run_suite,
    starter_run_suite_filename,
)
from tokenflood.util import get_date_str, get_run_name

log = logging.getLogger(__name__)

the_wave = f"""[blue]⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
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
    tokenflood_logger.addHandler(
        RichHandler(markup=True, highlighter=NullHighlighter(), keywords=[])
    )
    for handler in tokenflood_logger.handlers:
        handler.addFilter(global_warn_once_filter)


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

    # Initialization
    init_cmd_parser = subparsers.add_parser(
        "init",
        help="[blue]Create starter files for run suite and endpoint specifications.[/]",
    )
    init_cmd_parser.set_defaults(func=create_starter_files)

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
    date_str = get_date_str()
    run_name = get_run_name(date_str, endpoint_spec)

    accepted_token_usage = check_token_usage_upfront(
        suite,
        args.autoaccept,
    )
    if not accepted_token_usage:
        log.info("Stopping because token usage was not accepted.")
        return

    run_folder = make_run_folder(run_name)
    log.info(f"Preparing run folder: [blue]{run_folder}[/]")

    endpoint_spec_file = os.path.join(run_folder, ENDPOINT_SPEC_FILE)
    log.info(f"Writing endpoint spec to: [blue]{endpoint_spec_file}[/]")
    write_pydantic_yaml(endpoint_spec_file, endpoint_spec)

    run_suite_file = os.path.join(run_folder, RUN_SUITE_FILE)
    log.info(f"Writing run suite to: [blue]{run_suite_file}[/]")
    write_pydantic_yaml(run_suite_file, suite)

    error_file = os.path.join(run_folder, ERROR_FILE)
    llm_requests_file = os.path.join(run_folder, LLM_REQUESTS_FILE)
    network_latency_file = os.path.join(run_folder, NETWORK_LATENCY_FILE)
    io_context = FileIOContext(llm_requests_file, network_latency_file, error_file)
    log.info("Starting load test")
    log.info(f"Streaming any errors to: [blue]{error_file}[/]")
    log.info(f"Streaming LLM request data to: [blue]{llm_requests_file}[/]")
    log.info(f"Streaming network latency data to: [blue]{network_latency_file}[/]")

    asyncio.run(run_suite(endpoint_spec, suite, io_context))
    log.info("Analyzing data.")
    llm_request_data = pd.read_csv(llm_requests_file)
    ping_data = pd.read_csv(network_latency_file)
    summary_file = os.path.join(run_folder, SUMMARY_FILE)
    latency_graph_file = os.path.join(run_folder, LATENCY_GRAPH_FILE)
    summary = create_summary(suite, endpoint_spec, llm_request_data, ping_data)
    write_pydantic_yaml(summary_file, summary)
    warn_relative_error(summary)
    title = make_super_title(suite, endpoint_spec, date_str, summary)
    visualize_percentiles_across_request_rates(title, summary, latency_graph_file)
    io_context.close()
    log.info("Done.")


def warn_relative_error(summary: RunSummary):
    if abs(summary.relative_input_token_error) > WARNING_LIMIT:
        sign = "more" if summary.relative_input_token_error > 0 else "less"
        log.warning(
            f"On average, the prompts had {abs(int(summary.relative_input_token_error * 100))}% {sign} tokens than expected. The observed latencies might not be representative."
        )

    if abs(summary.relative_output_token_error) > WARNING_LIMIT:
        sign = "more" if summary.relative_output_token_error > 0 else "less"
        log.warning(
            f"On average, the generated texts had {abs(int(summary.relative_output_token_error * 100))}% {sign} tokens than expected. The observed latencies might not be representative."
        )


def main():
    load_dotenv(".env")
    configure_logging()
    args = parse_args(sys.argv[1:])
    log.info(
        the_wave,
    )
    args.func(args)
