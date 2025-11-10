from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_suite import HeuristicRunSuite
from tokenflood.models.run_summary import LoadResult, RunSummary
from tokenflood.models.util import numeric
from tokenflood.util import calculate_relative_error

PHASE_FIELD = "requests_per_second_phase"


def get_phase_data(data: pd.DataFrame, phase: float) -> pd.DataFrame:
    return data[data[PHASE_FIELD] == phase]


def get_phases(data: pd.DataFrame) -> List[float]:
    return list(pd.unique(data[PHASE_FIELD]))


def make_super_title(
    run_suite: HeuristicRunSuite,
    endpoint_spec: EndpointSpec,
    date_str: str,
    summary: RunSummary,
) -> str:
    return f"run suite: {run_suite.name}\nmodel: {endpoint_spec.provider_model_str}\ndatetime: {date_str}\n⌀ input/prefix/output tokens: {summary.mean_measured_input_tokens}/{summary.mean_expected_prefix_tokens}/{summary.mean_measured_output_tokens}"


def visualize_percentiles_across_request_rates(
    title: str, run_summary: RunSummary, filename: str
):
    phases = [lr.requests_per_second for lr in run_summary.load_results]
    percentiles = (
        list(run_summary.load_results[0].percentile_latency.keys())
        if len(run_summary.load_results)
        else []
    )
    for percentile in percentiles:
        y = [lr.percentile_latency[percentile] for lr in run_summary.load_results]
        plt.plot(phases, y, marker="o", markersize=3, label=f"{percentile} latency")

    avg_latencies = [lr.mean_request_latency for lr in run_summary.load_results]
    plt.plot(phases, avg_latencies, marker="o", markersize=3, label="mean latency")
    ping_latencies = [lr.mean_network_latency for lr in run_summary.load_results]
    plt.plot(
        phases, ping_latencies, marker="o", markersize=3, label="mean network latency"
    )
    plt.subplots_adjust(top=0.75)
    plt.xlabel("Requests per Second")
    plt.ylabel("Latency in ms")
    plt.suptitle(title, ha="left", x=0.125)
    plt.title("Latency percentiles across request rates")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def create_summary(
    run_suite: HeuristicRunSuite,
    endpoint_spec: EndpointSpec,
    llm_request_data: pd.DataFrame,
    ping_data: pd.DataFrame,
) -> RunSummary:
    total_num_requests = len(llm_request_data)
    if total_num_requests == 0:
        return RunSummary.create_empty(run_suite.name, endpoint_spec.provider_model_str)
    load_results = []
    phases = get_phases(llm_request_data)
    for phase in phases:
        phase_llm_request_data = get_phase_data(llm_request_data, phase)
        phase_ping_data = get_phase_data(ping_data, phase)
        percentiles = {}
        for percentile in run_suite.percentiles:
            percentiles[f"p{percentile}"] = get_percentile_float(
                list(phase_llm_request_data["latency"]), percentile
            )
        load_results.append(
            LoadResult(
                requests_per_second=float(phase),
                mean_request_latency=round(
                    float(np.average(phase_llm_request_data["latency"])), 2
                ),
                mean_network_latency=round(
                    float(np.average(phase_ping_data["latency"])), 2
                ),
                percentile_latency=percentiles,
            )
        )

    return RunSummary(
        run_suite=run_suite.name,
        endpoint=endpoint_spec.provider_model_str,
        total_num_requests=total_num_requests,
        mean_expected_input_tokens=int(
            np.average(llm_request_data["expected_input_tokens"])
        ),
        mean_measured_input_tokens=int(
            np.average(llm_request_data["measured_input_tokens"])
        ),
        mean_expected_output_tokens=int(
            np.average(llm_request_data["expected_output_tokens"])
        ),
        mean_measured_output_tokens=int(
            np.average(llm_request_data["measured_output_tokens"])
        ),
        mean_expected_prefix_tokens=int(
            np.average(llm_request_data["expected_prefix_tokens"])
        ),
        mean_measured_prefix_tokens=int(
            np.average(llm_request_data["measured_prefix_tokens"])
        ),
        relative_input_token_error=calculate_relative_error(
            list(llm_request_data["measured_input_tokens"]),
            list(llm_request_data["expected_input_tokens"]),
        ),
        relative_prefix_token_error=calculate_relative_error(
            list(llm_request_data["measured_prefix_tokens"]),
            list(llm_request_data["expected_prefix_tokens"]),
        ),
        relative_output_token_error=calculate_relative_error(
            list(llm_request_data["measured_output_tokens"]),
            list(llm_request_data["expected_output_tokens"]),
        ),
        load_results=load_results,
    )


def get_percentile_float(seq: Sequence[numeric], percentile: int) -> float:
    value = 0.0
    if len(seq) == 1:
        value = seq[0]
    elif len(seq) > 1:
        value = round(float(np.percentile(seq, percentile)), 2)
    return value
