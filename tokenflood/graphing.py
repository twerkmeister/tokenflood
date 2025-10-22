from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from tokenflood.models.run_data import RunData
from tokenflood.models.run_suite import HeuristicRunSuite


def visualize_percentiles_across_request_rates(
    run_suite: HeuristicRunSuite, run_suite_data: List[RunData], filename: str
):
    x = run_suite.requests_per_second_rates
    for percentile in run_suite.percentiles:
        y = [
            run_data.results.get_latency_percentile(percentile)
            for run_data in run_suite_data
        ]

        plt.plot(x, y, label=f"p{percentile}")

    plt.xlabel("Requests per Second")
    plt.ylabel("Latency in ms")
    plt.title(run_suite.name)
    plt.legend()
    plt.savefig(filename)


def write_out_raw_data_points(run_data_list: List[RunData], filename: str):
    run_data_dfs = [data.as_dataframe() for data in run_data_list]
    run_data_df = pd.concat(run_data_dfs)
    run_data_df.to_csv(filename)
