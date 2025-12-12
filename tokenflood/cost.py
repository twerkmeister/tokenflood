import logging

from tokenflood.models.budget import Budget
from tokenflood.models.token_cost_aware import TokenCostAware


log = logging.getLogger(__name__)


def check_token_usage_upfront(
    costly_operation: TokenCostAware,
    budget: Budget,
    proceed: bool,
) -> bool:
    estimated_input_tokens, estimated_output_tokens = (
        costly_operation.get_input_output_token_cost()
    )
    log.info("Checking estimated token usage for the run:")
    input_token_color = get_limit_color(estimated_input_tokens, budget.input_tokens)
    output_token_color = get_limit_color(estimated_output_tokens, budget.output_tokens)
    log.info(
        f"Estimated input tokens / configured max input tokens: "
        f"[{input_token_color}]{estimated_input_tokens}[/] / [blue]{budget.input_tokens}[/]"
    )
    log.info(
        f"Estimated output tokens / configured max output tokens: "
        f"[{output_token_color}]{estimated_output_tokens}[/] / [blue]{budget.output_tokens}[/]"
    )
    if (
        estimated_input_tokens > budget.input_tokens
        or estimated_output_tokens > budget.output_tokens
    ):
        log.info("[red]Estimated tokens beyond configured budget. Aborting the run.[/]")
        log.info(
            "Increase the maximum tokens you are willing to spend by setting the variables "
            "[red]budget.input_tokens[/] and [red]budget.output_tokens[/] to a higher value "
            f"in your {costly_operation.__class__.__name__} file."
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
