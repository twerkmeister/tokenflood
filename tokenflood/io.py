import os
from typing import Callable, Type, TypeVar

import yaml
from pydantic import BaseModel

from tokenflood.constants import RESULTS_FOLDER
from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_suite import HeuristicRunSuite

T = TypeVar("T", bound=BaseModel)


def read_pydantic_yaml(clazz: Type[T]) -> Callable[[str], T]:
    def read_clazz(filename: str) -> T:
        with open(filename) as f:
            data = yaml.safe_load(f)
        return clazz(**data)

    return read_clazz


def read_endpoint_spec(filename: str) -> EndpointSpec:
    return read_pydantic_yaml(EndpointSpec)(filename)


def read_run_suite(filename: str) -> HeuristicRunSuite:
    return read_pydantic_yaml(HeuristicRunSuite)(filename)


def make_run_folder(run_name: str) -> str:
    run_folder = os.path.join(RESULTS_FOLDER, run_name)
    os.makedirs(run_folder)
    return run_folder
