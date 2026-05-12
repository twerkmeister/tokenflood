from __future__ import annotations

import functools
import os
from typing import Callable, Optional

import pandas as pd
import logging

from tokenflood.constants import (
    ERROR_FILE,
    LLM_REQUESTS_FILE,
    NETWORK_LATENCY_FILE,
    ENDPOINT_SPEC_FILE,
    LOAD_TEST_SPEC_FILE,
    OBSERVATION_SPEC_FILE,
)
from tokenflood.io import (
    is_load_test_result_folder,
    is_observation_result_folder,
    read_file,
)

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=100)
def read_dataframe(path: str, csv_file: str = "") -> pd.DataFrame:
    if csv_file:
        path = os.path.join(path, csv_file)
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(str(e))
    return df


def get_runs(
    folder: str, predicate: Optional[Callable[[str], bool]] = None
) -> list[str]:
    if not os.path.isdir(folder):
        return []
    runs = sorted(os.listdir(folder), reverse=True)
    runs = [os.path.join(folder, run) for run in runs]
    if predicate:
        runs = [run for run in runs if predicate(run)]
    return [os.path.basename(run) for run in runs]


def get_load_test_runs(folder: str) -> list[str]:
    return get_runs(folder, is_load_test_result_folder)


def get_observation_runs(folder: str) -> list[str]:
    return get_runs(folder, is_observation_result_folder)


def get_error_dataframe(folder: str) -> pd.DataFrame:
    return read_dataframe(folder, ERROR_FILE)


def get_endpoint_spec_file(run_folder: str) -> str:
    return read_file(os.path.join(run_folder, ENDPOINT_SPEC_FILE))


def get_run_spec_file(run_folder: str) -> str:
    return read_file(os.path.join(run_folder, LOAD_TEST_SPEC_FILE))


def get_observation_spec_file(run_folder: str) -> str:
    return read_file(os.path.join(run_folder, OBSERVATION_SPEC_FILE))


def get_llm_request_dataframe(folder: str) -> pd.DataFrame:
    return read_dataframe(folder, LLM_REQUESTS_FILE)


def get_network_dataframe(folder: str) -> pd.DataFrame:
    return read_dataframe(folder, NETWORK_LATENCY_FILE)
