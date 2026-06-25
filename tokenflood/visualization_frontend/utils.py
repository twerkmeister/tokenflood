import os
from functools import lru_cache, wraps

from tokenflood.constants import LLM_REQUESTS_FILE, ERROR_FILE, NETWORK_LATENCY_FILE


def get_file_size(path) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return -1


def cache_if_run_data_stayed_the_same(func):
    @lru_cache(maxsize=None)
    def cached_wrapper(
        filename, request_file_size, error_file_size, latency_file_size, *args, **kwargs
    ):
        return func(filename, *args, **kwargs)

    @wraps(func)
    def wrapper(run_folder, *args, **kwargs):
        request_file = os.path.join(run_folder, LLM_REQUESTS_FILE)
        error_file = os.path.join(run_folder, ERROR_FILE)
        latency_file = os.path.join(run_folder, NETWORK_LATENCY_FILE)
        request_file_size = get_file_size(request_file)
        error_file_size = get_file_size(error_file)
        latency_file_size = get_file_size(latency_file)

        # Pass the size into the cache key automatically
        return cached_wrapper(
            run_folder,
            request_file_size,
            error_file_size,
            latency_file_size,
            *args,
            **kwargs,
        )

    return wrapper


def cache_if_csv_stayed_the_same(func):
    # Base cache that will store the actual results
    @lru_cache(maxsize=None)
    def cached_wrapper(path, file_size, csv_file="", *args, **kwargs):
        return func(path, csv_file, *args, **kwargs)

    @wraps(func)
    def wrapper(path, csv_file="", *args, **kwargs):
        real_path = path
        if csv_file:
            real_path = os.path.join(path, csv_file)
        file_size = get_file_size(real_path)
        # Pass the size into the cache key automatically
        return cached_wrapper(path, file_size, csv_file, *args, **kwargs)

    return wrapper
