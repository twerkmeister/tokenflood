import asyncio
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import logging
import numpy as np
from aiohttp import ClientSession
from litellm import acompletion
from litellm.types.utils import ModelResponse, Usage
from tqdm import tqdm

from tokenflood.constants import MAX_INPUT_TOKENS_ENV_VAR, MAX_OUTPUT_TOKENS_ENV_VAR
from tokenflood.heuristic import (
    create_heuristic_messages,
    heuristic_tasks,
    heuristic_token_sets,
)
from tokenflood.io import error_to_str
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.messages import MessageList, create_message_list_from_prompt
from tokenflood.models.results import Results
from tokenflood.models.run_data import RunData
from tokenflood.models.run_spec import HeuristicRunSpec, RunSpec
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.token_set import TokenSet

log = logging.getLogger(__name__)


@dataclass
class RunState:
    error: Optional[str] = None


def set_error_state(run_state: RunState) -> Callable[[asyncio.Task], None]:
    """A callback to help stop a run on the first error."""

    def on_done(task: asyncio.Task):
        if task.exception():
            run_state.error = error_to_str(task.exception())

    return on_done


def make_empty_response() -> ModelResponse:
    """Create an empty ModelResponse object as a placeholder for skipped requests.

    Requests are skipped once an error happens during any requests."""
    return ModelResponse(
        choices=[{"message": {"content": ""}}],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        _response_ms=0,
    )


def create_schedule(run_spec: RunSpec) -> List[float]:
    """Create a randomized schedule with a guaranteed total length."""
    pauses = np.random.exponential(
        1 / run_spec.requests_per_second, size=run_spec.total_num_requests
    )
    total_length = pauses.sum()
    pauses = pauses / (total_length / run_spec.test_length_in_seconds)
    return list(pauses)


def collect_results(
    message_lists: List[MessageList],
    expected_input_lengths: List[int],
    expected_prefix_lengths: List[int],
    expected_output_lengths: List[int],
    model_responses: List[ModelResponse],
) -> Results:
    usages: List[Usage] = [mr.usage for mr in model_responses]  # type: ignore[attr-defined]
    return Results(
        prompts=[ml[0]["content"] for ml in message_lists],
        generated_texts=[mr.choices[0]["message"]["content"] for mr in model_responses],
        latencies=tuple([int(mr._response_ms) for mr in model_responses]),  # type: ignore[attr-defined]
        expected_input_lengths=tuple(expected_input_lengths),
        expected_prefix_lengths=tuple(expected_prefix_lengths),
        expected_output_lengths=tuple(expected_output_lengths),
        measured_input_lengths=tuple([usage.prompt_tokens for usage in usages]),
        measured_prefix_lengths=tuple(
            [
                usage.prompt_tokens_details.cached_tokens or 0
                if usage.prompt_tokens_details
                else 0
                for usage in usages
            ]
        ),
        measured_output_lengths=tuple([usage.completion_tokens for usage in usages]),
    )


async def run_heuristic_test(
    run_suite_name: str,
    phase: int,
    run_spec: HeuristicRunSpec,
    endpoint_spec: EndpointSpec,
    token_set: Optional[TokenSet] = None,
    task: Optional[HeuristicTask] = None,
) -> RunData:
    test_name = f"Run suite {run_suite_name} phase {phase}: {run_spec.requests_per_second:.2f} requests/s"
    token_set = token_set or heuristic_token_sets[0]
    task = task or heuristic_tasks[0]
    schedule = create_schedule(run_spec)
    prompt_lengths, prefix_lengths, output_lengths = run_spec.sample()
    message_lists = create_heuristic_messages(
        prompt_lengths, prefix_lengths, token_set, task
    )
    model_responses, error = await run_test(
        test_name, schedule, message_lists, output_lengths, endpoint_spec
    )
    results = collect_results(
        message_lists, prompt_lengths, prefix_lengths, output_lengths, model_responses
    )
    return RunData(
        run_spec=run_spec, responses=model_responses, results=results, error=error
    )


def mend_responses(
    responses_and_errors: List[Union[ModelResponse, BaseException]],
    target_num_responses: int,
) -> List[ModelResponse]:
    """Replace failed and skipped requests with empty responses."""
    # replace exceptions with model responses
    mended_responses: List[ModelResponse] = [
        r if isinstance(r, ModelResponse) else make_empty_response()
        for r in responses_and_errors
    ]
    # fill responses for skipped responses with empty responses
    num_skipped_responses = target_num_responses - len(mended_responses)
    skipped_responses = [make_empty_response() for _ in range(num_skipped_responses)]
    return mended_responses + skipped_responses


async def run_test(
    name: str,
    schedule: List[float],
    message_lists: List[MessageList],
    num_generation_tokens: List[int],
    endpoint_spec: EndpointSpec,
) -> Tuple[List[ModelResponse], Optional[str]]:
    request_tasks: List[asyncio.Task] = []
    log.info("Warming up for the new phase.")
    client_session = ClientSession()
    await warm_up_session(endpoint_spec, client_session)
    state = RunState()
    for i in tqdm(range(len(schedule)), desc=name):
        request_task = asyncio.create_task(
            send_llm_request(endpoint_spec, message_lists[i], num_generation_tokens[i], client_session)
        )
        request_tasks.append(request_task)
        request_task.add_done_callback(set_error_state(state))
        await asyncio.sleep(schedule[i])
        if state.error:
            break

    log.info("Waiting for all requests to come back.")
    responses_and_errors = await asyncio.gather(*request_tasks, return_exceptions=True)
    responses = mend_responses(responses_and_errors, len(schedule))
    await client_session.close()
    log.info("Finished the phase.")
    return responses, state.error

async def warm_up_session(endpoint_spec: EndpointSpec, client_session: ClientSession):
    message_list = create_message_list_from_prompt("Hello!")
    return await send_llm_request(endpoint_spec, message_list, 1, client_session)


async def send_llm_request(
    endpoint_spec: EndpointSpec, messages: MessageList, num_generation_tokens: int, client_session: ClientSession
) -> ModelResponse:
    return await acompletion(
        model=endpoint_spec.provider_model_str,
        messages=messages,
        max_tokens=num_generation_tokens,
        base_url=endpoint_spec.base_url,
        api_key=endpoint_spec.api_key_env_var,
        deployment_id=endpoint_spec.deployment,
        extra_headers=endpoint_spec.extra_headers,
        max_retries=0,
        shared_session=client_session
    )


async def run_suite(
    endpoint_spec: EndpointSpec, suite: HeuristicRunSuite
) -> List[RunData]:
    run_specs = suite.create_run_specs()
    run_suite_data = []
    for phase, run_spec in enumerate(run_specs):
        run_data = await run_heuristic_test(suite.name, phase+1, run_spec, endpoint_spec)
        run_suite_data.append(run_data)
        if run_data.error:
            log.error(f"Ending run due to error: {run_data.error}")
            break
    return run_suite_data


def estimate_token_usage(suite: HeuristicRunSuite) -> Tuple[int, int]:
    """Estimate total token usage based on the run suite parameters.

    Specifically: requests per seconds, length of test, load types."""
    total_input_tokens = 0
    total_output_tokens = 0
    for run_spec in suite.create_run_specs():
        input_tokens, _, output_tokens = run_spec.sample()
        total_input_tokens += sum(input_tokens)
        total_output_tokens += sum(output_tokens)
    return total_input_tokens, total_output_tokens


def check_token_usage_upfront(
    suite: HeuristicRunSuite,
    max_input_tokens: int,
    max_output_tokens: int,
    proceed: bool,
) -> bool:
    estimated_input_tokens, estimated_output_tokens = estimate_token_usage(suite)
    log.info("Checking estimated token usage for the run:")
    input_token_color = get_limit_color(estimated_input_tokens, max_input_tokens)
    output_token_color = get_limit_color(estimated_output_tokens, max_output_tokens)
    log.info(
        f"Estimated input tokens / configured max input tokens: "
        f"[{input_token_color}]{estimated_input_tokens}[/] / [blue]{max_input_tokens}[/]"
    )
    log.info(
        f"Estimated output tokens / configured max output tokens: "
        f"[{output_token_color}]{estimated_output_tokens}[/] / [blue]{max_output_tokens}[/]"
    )
    if (
        estimated_input_tokens > max_input_tokens
        or estimated_output_tokens > max_output_tokens
    ):
        log.info(
            "[red]Estimated tokens beyond configured maximum. Aborting the run.[/]"
        )
        log.info(
            "Increase the maximum tokens you are willing to spend via the env vars "
            f"[red]{MAX_INPUT_TOKENS_ENV_VAR}[/] and [red]{MAX_OUTPUT_TOKENS_ENV_VAR}[/]"
        )
        return False

    if proceed:
        log.info("Token usage [blue]auto-accepted[/blue]")
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


def get_limit_color(n: int, target: int) -> str:
    if n > target:
        return "red"
    return "blue"
