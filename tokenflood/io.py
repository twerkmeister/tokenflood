import os
from typing import Callable, List, Optional, Type, TypeVar
import yaml
from pydantic import BaseModel

from tokenflood.constants import RESULTS_FOLDER
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_suite import HeuristicRunSuite

T = TypeVar("T", bound=BaseModel)


def read_pydantic_yaml(class_type: Type[T]) -> Callable[[str], T]:
    def read_class_type(filename: str) -> T:
        with open(filename) as f:
            data = yaml.safe_load(f)
        return class_type(**data)

    return read_class_type


def read_pydantic_yaml_list(class_type: Type[T]) -> Callable[[str], List[T]]:
    def read_class_type_list(filename: str) -> List[T]:
        with open(filename) as f:
            list_data = yaml.safe_load(f)
        return [class_type(**data) for data in list_data]

    return read_class_type_list


def write_pydantic_yaml(filename: str, o: T) -> None:
    with open(filename, "w") as f:
        yaml.safe_dump(o.model_dump(), f, sort_keys=False)


def write_pydantic_yaml_list(filename: str, objects: List[T]) -> None:
    with open(filename, "w") as f:
        yaml.safe_dump([o.model_dump() for o in objects], f, sort_keys=False)


def read_endpoint_spec(filename: str) -> EndpointSpec:
    return read_pydantic_yaml(EndpointSpec)(filename)


def read_run_suite(filename: str) -> HeuristicRunSuite:
    return read_pydantic_yaml(HeuristicRunSuite)(filename)


def make_run_folder(run_name: str) -> str:
    run_folder = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(run_folder)
    return run_folder


def add_suffix_to_file_name(filename: str, suffix: str) -> str:
    filename, ext = os.path.splitext(filename)
    return f"{filename}{suffix}{ext}"


def get_first_available_filename_like(filename: str) -> str:
    """Find the first version of a file name that isn't taken yet.

    Adds a suffix to find another available version."""
    suffix = 1
    current_name = filename
    while os.path.exists(current_name):
        current_name = add_suffix_to_file_name(filename, f"_{suffix:02d}")
        suffix += 1
    return current_name


def list_dir_relative(folder_name: str) -> List[str]:
    """List dir while preserving the relative path name."""
    return [os.path.join(folder_name, f) for f in os.listdir(folder_name)]


def write_file(filename: str, s: str):
    with open(filename, "w") as f:
        f.write(s)


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def error_to_str(e: Optional[BaseException]) -> Optional[str]:
    # return "\n".join(traceback.format_exception(e))
    if e:
        return str(e)
    else:
        return None
