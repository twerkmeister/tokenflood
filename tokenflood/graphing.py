from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tokenflood.io import write_file
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_data import RunData
from tokenflood.models.run_suite import HeuristicRunSuite


def visualize_percentiles_across_request_rates(
    run_suite: HeuristicRunSuite, run_suite_data: List[RunData], filename: str
):
    x = [data.run_spec.requests_per_second for data in run_suite_data]
    for percentile in run_suite.percentiles:
        y = [
            run_data.results.get_latency_percentile(percentile)
            for run_data in run_suite_data
        ]
        plt.plot(x, y, marker='o', markersize=3, label=f"p{percentile}")

    ping_latencies = [data.average_network_latency() for data in run_suite_data]
    plt.plot(x, ping_latencies, marker='o', markersize=3, label="network latency")

    plt.xlabel("Requests per Second")
    plt.ylabel("Latency in ms")
    plt.suptitle(f"Run suite: {run_suite.name}")
    plt.title("Latency percentiles across request rates")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def write_out_raw_data_points(run_data_list: List[RunData], filename: str):
    if len(run_data_list) > 0:
        run_data_dfs = [data.as_dataframe() for data in run_data_list]
        run_data_df = pd.concat(run_data_dfs)
        run_data_df.to_csv(filename, index=False)


def write_out_error(run_data_list: List[RunData], filename: str):
    for run_data in run_data_list:
        if run_data.error:
            write_file(filename, run_data.error)
            break


def write_out_summary(
    run_suite: HeuristicRunSuite,
    endpoint_spec: EndpointSpec,
    run_data_list: List[RunData],
    filename: str,
):
    total_num_requests = sum([len(rd.responses) for rd in run_data_list])
    summary_data = {
        "run_suite": run_suite.name,
        "endpoint": endpoint_spec.provider_model_str,
        "total_num_requests": total_num_requests,
    }
    load_results = []
    for rd in run_data_list:
        load_result = {
            "requests_per_second": rd.run_spec.requests_per_second,
            "mean_latency": round(float(np.average(rd.results.latencies)), 2),
            "mean_network_latency": rd.average_network_latency(),
            "relative_input_token_error": rd.results.get_relative_input_length_error(),
            "relative_output_token_error": rd.results.get_relative_output_length_error(),
            "relative_prefix_token_error": rd.results.get_relative_prefix_length_error(),
        }
        for percentile in run_suite.percentiles:
            load_result[f"p{percentile}"] = rd.results.get_latency_percentile(
                percentile
            )

        load_results.append(load_result)
    summary_data["load_results"] = load_results
    write_file(filename, yaml.safe_dump(summary_data, sort_keys=False))
