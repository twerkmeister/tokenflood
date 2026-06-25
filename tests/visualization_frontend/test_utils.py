import os
import shutil

from tokenflood.constants import LLM_REQUESTS_FILE, ERROR_FILE, NETWORK_LATENCY_FILE
from tokenflood.visualization_frontend.utils import cache_if_run_data_stayed_the_same


def add_new_line(file: str):
    assert os.path.isfile(file)
    with open(file, "a") as f:
        f.write("new line\n")


def test_data_caching_when_data_changes(
    unique_temporary_folder, results_folder, monkeypatch
):
    monkeypatch.chdir(unique_temporary_folder)
    copy_of_results_folder = os.path.join(unique_temporary_folder, "results")
    shutil.copytree(results_folder, copy_of_results_folder)
    assert "results" in os.listdir(unique_temporary_folder)

    count = 0

    @cache_if_run_data_stayed_the_same
    def increase_counter(result_folder: str) -> int:
        nonlocal count
        count += 1
        return count

    folder_1 = os.path.join(copy_of_results_folder, "load_test_results")
    folder_2 = os.path.join(copy_of_results_folder, "observation_results")
    assert os.path.isdir(folder_1)
    assert os.path.isdir(folder_2)

    assert count == 0
    # calling multiple times -> single increase then cached
    assert increase_counter(folder_1) == 1
    assert increase_counter(folder_1) == 1
    assert increase_counter(folder_1) == 1

    # calling with a new folder -> single increase then cached
    assert increase_counter(folder_2) == 2
    assert increase_counter(folder_2) == 2
    assert increase_counter(folder_2) == 2

    # going for the first one again -> cached
    assert increase_counter(folder_1) == 1

    # writing to folder 1 results
    folder_1_requests_data = os.path.join(folder_1, LLM_REQUESTS_FILE)
    add_new_line(folder_1_requests_data)

    # now the cache should not trigger
    assert increase_counter(folder_1) == 3
    assert increase_counter(folder_1) == 3

    folder_1_error_data = os.path.join(folder_1, ERROR_FILE)
    add_new_line(folder_1_error_data)

    # now the cache should not trigger
    assert increase_counter(folder_1) == 4
    assert increase_counter(folder_1) == 4

    folder_1_latency_data = os.path.join(folder_1, NETWORK_LATENCY_FILE)
    add_new_line(folder_1_latency_data)

    # now the cache should not trigger
    assert increase_counter(folder_1) == 5
    assert increase_counter(folder_1) == 5
