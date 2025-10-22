import os
from typing import Callable, List, Type, TypeVar

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
        yaml.safe_dump(o.model_dump(), f)


def write_pydantic_yaml_list(filename: str, objects: List[T]) -> None:
    with open(filename, "w") as f:
        yaml.safe_dump([o.model_dump() for o in objects], f)


def read_endpoint_spec(filename: str) -> EndpointSpec:
    return read_pydantic_yaml(EndpointSpec)(filename)


def read_run_suite(filename: str) -> HeuristicRunSuite:
    return read_pydantic_yaml(HeuristicRunSuite)(filename)


def make_run_folder(run_name: str) -> str:
    run_folder = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(run_folder)
    return run_folder
