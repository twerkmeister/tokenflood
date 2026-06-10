import argparse
import asyncio
import json
import os
import sys
from io import StringIO
from typing import List, Tuple, Callable, Coroutine, TypeVar, Any

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
    LOAD_TEST_SPEC_FILE,
)
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.messages import (
    create_message_list_from_prompt,
    get_input_output_prefix_token_lengths,
)
from tokenflood.models.run_specs.load_test_spec import LoadTestSpec
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.visualization_frontend.gradio import visualize_results
from tokenflood.io import (
    FileIOContext,
    get_first_available_filename_like,
    make_run_folder,
    read_endpoint_spec,
    write_pydantic_yaml,
    read_run_spec,
    IOContext,
    read_file,
    read_jsonl_messages,
)
from tokenflood.logging_utils import global_warn_once_filter
from tokenflood.networking import (
    patch_aiohttp_client_session,
    unpatch_aiohttp_client_session,
)
from tokenflood.observer import run_observation
from tokenflood.runner import run_load_test
from tokenflood.starter_pack import (
    starter_endpoint_spec_vllm,
    starter_model_id,
    starter_observation_spec,
    starter_run_suite,
)

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
        "run", help="[blue]Execute a load- or observation test.[/]"
    )
    run_cmd_parser.add_argument("run_spec", type=str)
    run_cmd_parser.add_argument("endpoint", type=str)
    run_cmd_parser.set_defaults(func=run)
    run_cmd_parser.add_argument(
        "-y",
        "--autoaccept",
        help="Auto accept run start.",
        action="store_true",
    )

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
        help="[blue]Create starter files for load testing and observation of endpoints.[/]",
    )
    init_cmd_parser.set_defaults(func=create_starter_files)

    # count
    count_cmd_parser = subparsers.add_parser(
        "count", help="[blue]Count tokens in a set of prompts.[/]"
    )
    count_cmd_parser.add_argument("prompt_file", type=str, nargs="+")
    count_cmd_parser.add_argument(
        "-f",
        "--format",
        choices=["text", "chat"],
        default="text",
        help="Format of the prompt files. Can be a files containing single prompts in text format (text) or a jsonl files with chat messages format.",
    )
    group = count_cmd_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Specify the endpoint spec to use for api-based token counting. Works only for openai, bedrock, azure, anthropic, gemini and vertex ai models. Others will fallback to local tiktoken.",
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Specify the huggingface model name whose tokenizer to use for token counting",
    )

    count_cmd_parser.set_defaults(func=count_prompt_tokens)

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
    available_load_test_spec_filename = get_first_available_filename_like(
        LOAD_TEST_SPEC_FILE
    )
    available_observation_spec_file = get_first_available_filename_like(
        OBSERVATION_SPEC_FILE
    )
    log.info(
        "Creating starter files for load, observation and endpoint specifications: \n"
        f"[green]* {available_load_test_spec_filename} [/]\n"
        f"[green]* {available_endpoint_spec_filename} [/]\n"
        f"[green]* {available_observation_spec_file} [/]"
    )

    write_pydantic_yaml(available_endpoint_spec_filename, starter_endpoint_spec_vllm)
    write_pydantic_yaml(available_load_test_spec_filename, starter_run_suite)
    write_pydantic_yaml(available_observation_spec_file, starter_observation_spec)

    log.info(
        "Inspect those files, boot up a vllm server with a tiny model using \n"
        f"[blue]vllm serve {starter_model_id}[/]\n"
        "and run a first load test against it using\n"
        f"[blue]tokenflood run {available_load_test_spec_filename} {available_endpoint_spec_filename}[/]\n"
        "or do a longer term observation using\n"
        f"[blue]tokenflood run {available_observation_spec_file} {available_endpoint_spec_filename}[/]"
    )


def run(args: argparse.Namespace):
    endpoint_spec = read_endpoint_spec(args.endpoint)
    run_spec = read_run_spec(args.run_spec)
    run_name = run_spec.get_run_name(endpoint_spec)

    confirm_start_run = confirm_starting_run(args.autoaccept)
    if not confirm_start_run:
        log.info("Stopping because starting confirmation was not given.")
        return

    run_folder = make_run_folder(run_name)
    log.info(f"Preparing results folder: [blue]{run_folder}[/]")

    endpoint_spec_file = os.path.join(run_folder, ENDPOINT_SPEC_FILE)
    log.info(f"Writing endpoint spec to: [blue]{endpoint_spec_file}[/]")
    write_pydantic_yaml(endpoint_spec_file, endpoint_spec)

    run_spec_file = os.path.join(run_folder, run_spec.run_spec_file)
    log.info(f"Writing run suite to: [blue]{run_spec_file}[/]")
    write_pydantic_yaml(run_spec_file, run_spec)

    error_file = os.path.join(run_folder, ERROR_FILE)
    llm_requests_file = os.path.join(run_folder, LLM_REQUESTS_FILE)
    network_latency_file = os.path.join(run_folder, NETWORK_LATENCY_FILE)
    io_context = FileIOContext(llm_requests_file, network_latency_file, error_file)
    log.info("Starting load test")
    log.info(f"Streaming any errors to: [blue]{error_file}[/]")
    log.info(f"Streaming LLM request data to: [blue]{llm_requests_file}[/]")
    log.info(f"Streaming network latency data to: [blue]{network_latency_file}[/]")
    test_procedure = get_test_procedure(run_spec)
    asyncio.run(test_procedure(endpoint_spec, run_spec, io_context))
    io_context.close()
    log.info("Done.")


def count_prompt_tokens(args: argparse.Namespace):
    endpoint_spec = read_endpoint_spec(args.endpoint) if args.endpoint else None

    message_lists = []
    for prompt_file in args.prompt_file:
        if args.format == "text":
            prompt = read_file(prompt_file)
            message_lists.extend([create_message_list_from_prompt(prompt)])
        else:
            message_lists.extend(read_jsonl_messages(prompt_file))

    input_lengths, output_lengths, prefix_lengths, common_prefix = asyncio.run(
        get_input_output_prefix_token_lengths(
            message_lists, endpoint_spec, args.tokenizer
        )
    )

    log.info("Input token lengths")
    log.info("===================")
    if len(input_lengths) > 0:
        log.info(f"number of input prompts: {len(input_lengths)}")
        log.info(f"min: {min(input_lengths)}")
        log.info(f"max: {max(input_lengths)}")
        log.info(f"avg: {sum(input_lengths) / len(input_lengths)}")
    else:
        log.info("no data")
    log.info("")

    log.info("Output token lengths")
    log.info("===================")
    if len(output_lengths) > 0:
        log.info(f"number of output prompts: {len(output_lengths)}")
        log.info(f"min: {min(output_lengths)}")
        log.info(f"max: {max(output_lengths)}")
        log.info(f"avg: {sum(output_lengths) / len(output_lengths)}")
    else:
        log.info("no data")
    log.info("")

    log.info("Common Prefix")
    log.info("===================")
    if prefix_lengths:
        log.info(f"prefix:\n{json.dumps(common_prefix, indent=4, ensure_ascii=False)}")
        log.info(f"length: {prefix_lengths[0]}")
    else:
        log.info("no data")


T = TypeVar("T", bound=LoadTestSpec | ObservationSpec)


def get_test_procedure(
    run_spec: T,
) -> Callable[[EndpointSpec, Any, IOContext], Coroutine]:
    if isinstance(run_spec, LoadTestSpec):
        return run_load_test
    elif isinstance(run_spec, ObservationSpec):
        return run_observation
    raise ValueError(
        f"Invalid run spec type: {type(run_spec)}. "
        f"Must be {LoadTestSpec.__name__} or {ObservationSpec.__name__}."
    )


def confirm_starting_run(proceed: bool = False) -> bool:
    if proceed:
        log.info("Run start [blue]auto-accepted[/blue]")
        return True

    response = "start_value"
    yes_answers = {"y", "yes"}
    no_answers = {"n", "no", ""}
    trials = 0
    while response not in yes_answers.union(no_answers) and trials < 3:
        response = input("Start the run? [y/N]: ")
        response = response.strip().lower()
        trials += 1
    return response in yes_answers


def main():
    load_dotenv(".env")
    configure_logging()
    patch_aiohttp_client_session()
    args = parse_args(sys.argv[1:])
    log.info(
        the_wave,
    )
    try:
        args.func(args)
    except KeyboardInterrupt:
        log.info("Stopping...")
    unpatch_aiohttp_client_session()


if __name__ == "__main__":
    main()
