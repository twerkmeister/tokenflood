import argparse
import asyncio
import os
import sys
from io import StringIO
from typing import List, Tuple

import gradio.routes
from rich import print
import logging

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from tokenflood import __version__

from dotenv import load_dotenv

from tokenflood.constants import (
    ENDPOINT_SPEC_FILE,
    ERROR_FILE,
    NETWORK_LATENCY_FILE,
    LLM_REQUESTS_FILE,
    OBSERVATION_SPEC_FILE,
    RUN_SUITE_FILE,
)
from tokenflood.gradio import visualize_results
from tokenflood.io import (
    FileIOContext,
    get_first_available_filename_like,
    make_run_folder,
    read_endpoint_spec,
    read_observation_spec,
    read_run_suite,
    write_pydantic_yaml,
)
from tokenflood.logging import global_warn_once_filter
from tokenflood.observer import run_observation
from tokenflood.runner import run_suite
from tokenflood.cost import check_token_usage_upfront
from tokenflood.starter_pack import (
    starter_endpoint_spec_vllm,
    starter_model_id,
    starter_observation_spec,
    starter_run_suite,
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
        "run", help="[blue]Execute a load test run suite and graph the results[/]"
    )
    run_cmd_parser.add_argument("run_suite", type=str)
    run_cmd_parser.add_argument("endpoint", type=str)
    run_cmd_parser.add_argument(
        "-y",
        "--autoaccept",
        help="Auto accept run start if tokens are within configured limits.",
        action="store_true",
    )
    run_cmd_parser.set_defaults(func=flood_endpoint)

    # OBSERVE
    observe_cmd_parser = subparsers.add_parser(
        "observe", help="[blue]observe an endpoint over a longer period of time.[/]"
    )
    observe_cmd_parser.add_argument("observation_spec", type=str)
    observe_cmd_parser.add_argument("endpoint", type=str)
    observe_cmd_parser.add_argument(
        "-y",
        "--autoaccept",
        help="Auto accept run start if tokens are within configured limits.",
        action="store_true",
    )
    observe_cmd_parser.set_defaults(func=observe_endpoint)

    # Visualize
    viz_cmd_parser = subparsers.add_parser(
        "viz", help="[blue]visualize the results files.[/]"
    )
    viz_cmd_parser.add_argument(
        "results_folder", type=str, nargs="?", default="./results"
    )
    viz_cmd_parser.set_defaults(func=start_visualization)

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


def start_visualization(
    args: argparse.Namespace, keep_running: bool = True, go_to_browser: bool = True
) -> Tuple[gradio.routes.App, str]:
    return visualize_results(args.results_folder, keep_running, go_to_browser)


def create_starter_files(args: argparse.Namespace):
    available_endpoint_spec_filename = get_first_available_filename_like(
        ENDPOINT_SPEC_FILE
    )
    available_run_suite_filename = get_first_available_filename_like(RUN_SUITE_FILE)
    available_observation_spec_file = get_first_available_filename_like(
        OBSERVATION_SPEC_FILE
    )
    log.info(
        "Creating starter files for run suite, observation and endpoint specifications: \n"
        f"[green]* {available_run_suite_filename} [/]\n"
        f"[green]* {available_endpoint_spec_filename} [/]\n"
        f"[green]* {available_observation_spec_file} [/]"
    )

    write_pydantic_yaml(available_endpoint_spec_filename, starter_endpoint_spec_vllm)
    write_pydantic_yaml(available_run_suite_filename, starter_run_suite)
    write_pydantic_yaml(available_observation_spec_file, starter_observation_spec)

    log.info(
        "Inspect those files, boot up a vllm server with a tiny model using \n"
        f"[blue]vllm serve {starter_model_id}[/]\n"
        "and run a first load test against it using\n"
        f"[blue]tokenflood run {available_run_suite_filename} {available_endpoint_spec_filename}[/]\n"
        "or do a longer term observation using\n"
        f"[blue]tokenflood observe {available_observation_spec_file} {available_endpoint_spec_filename}[/]"
    )


def flood_endpoint(args: argparse.Namespace):
    endpoint_spec = read_endpoint_spec(args.endpoint)
    suite = read_run_suite(args.run_suite)
    date_str = get_date_str()
    run_name = get_run_name(date_str, "flood", suite.name, endpoint_spec)

    accepted_token_usage = check_token_usage_upfront(
        suite,
        suite.budget,
        args.autoaccept,
    )
    if not accepted_token_usage:
        log.info("Stopping because token usage was not accepted.")
        return

    run_folder = make_run_folder(run_name)
    log.info(f"Preparing results folder: [blue]{run_folder}[/]")

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
    io_context.close()
    log.info("Done.")


def observe_endpoint(args: argparse.Namespace):
    endpoint_spec = read_endpoint_spec(args.endpoint)
    observation_spec = read_observation_spec(args.observation_spec)
    date_str = get_date_str()
    run_name = get_run_name(date_str, "observe", observation_spec.name, endpoint_spec)

    accepted_token_usage = check_token_usage_upfront(
        observation_spec,
        observation_spec.budget,
        args.autoaccept,
    )
    if not accepted_token_usage:
        log.info("Stopping because token usage was not accepted.")
        return

    run_folder = make_run_folder(run_name)
    log.info(f"Preparing results folder: [blue]{run_folder}[/]")

    endpoint_spec_file = os.path.join(run_folder, ENDPOINT_SPEC_FILE)
    log.info(f"Writing endpoint spec to: [blue]{endpoint_spec_file}[/]")
    write_pydantic_yaml(endpoint_spec_file, endpoint_spec)

    run_suite_file = os.path.join(run_folder, OBSERVATION_SPEC_FILE)
    log.info(f"Writing observation spec to: [blue]{run_suite_file}[/]")
    write_pydantic_yaml(run_suite_file, observation_spec)

    error_file = os.path.join(run_folder, ERROR_FILE)
    llm_requests_file = os.path.join(run_folder, LLM_REQUESTS_FILE)
    network_latency_file = os.path.join(run_folder, NETWORK_LATENCY_FILE)
    io_context = FileIOContext(llm_requests_file, network_latency_file, error_file)
    log.info("Starting observation test")
    log.info(f"Streaming any errors to: [blue]{error_file}[/]")
    log.info(f"Streaming LLM request data to: [blue]{llm_requests_file}[/]")
    log.info(f"Streaming network latency data to: [blue]{network_latency_file}[/]")

    asyncio.run(run_observation(endpoint_spec, observation_spec, io_context))
    io_context.close()
    log.info("Done.")


def main():
    load_dotenv(".env")
    configure_logging()
    args = parse_args(sys.argv[1:])
    log.info(
        the_wave,
    )
    try:
        args.func(args)
    except KeyboardInterrupt:
        log.info("Stopping...")
