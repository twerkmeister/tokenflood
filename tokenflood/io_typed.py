from typing import TypeVar, Type, Callable, List, Iterable, Any

import yaml
from pydantic import BaseModel, TypeAdapter

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.run_specs.load_test_spec import LoadTestSpec
from tokenflood.models.run_specs.observation_spec import ObservationSpec
from tokenflood.models.run_specs.typing import SpecificRunSpec

T = TypeVar("T", bound=BaseModel)


def create_from_basemodel_or_type_adapter(
    data: dict, class_type: Type[T] | TypeAdapter[T]
) -> T:
    if isinstance(class_type, type) and issubclass(class_type, BaseModel):
        return class_type(**data)
    elif isinstance(class_type, TypeAdapter):
        return class_type.validate_python(data)
    else:
        raise ValueError


def read_pydantic_yaml(class_type: Type[T] | TypeAdapter[T]) -> Callable[[str], T]:
    def read_class_type(filename: str) -> T:
        with open(filename) as f:
            data = yaml.safe_load(f)
        return create_from_basemodel_or_type_adapter(data, class_type)

    return read_class_type


def read_pydantic_yaml_list(
    class_type: Type[T] | TypeAdapter[T],
) -> Callable[[str], List[T]]:
    def read_class_type_list(filename: str) -> List[T]:
        with open(filename) as f:
            list_data = yaml.safe_load(f)
        return [
            create_from_basemodel_or_type_adapter(data, class_type)
            for data in list_data
        ]

    return read_class_type_list


class CustomDumper(yaml.SafeDumper):
    def represent_sequence(
        self, tag: str, sequence: Iterable[Any], flow_style: bool | None = None
    ):
        # making sure an iterator argument is not exhausted
        sequence = list(sequence)
        is_simple = all(isinstance(item, (int, float, str, bool)) for item in sequence)
        return super().represent_sequence(tag, sequence, flow_style=is_simple)


def write_pydantic_yaml(filename: str, o: T) -> None:
    with open(filename, "w") as f:
        yaml.dump(
            o.model_dump(),
            f,
            sort_keys=False,
            indent=2,
            Dumper=CustomDumper,
            default_flow_style=False,
        )


def write_pydantic_yaml_list(filename: str, objects: List[T]) -> None:
    with open(filename, "w") as f:
        yaml.dump(
            [o.model_dump() for o in objects],
            f,
            sort_keys=False,
            Dumper=CustomDumper,
            indent=2,
            default_flow_style=False,
        )


def read_endpoint_spec(filename: str) -> EndpointSpec:
    return read_pydantic_yaml(EndpointSpec)(filename)


def read_run_spec(filename: str) -> SpecificRunSpec:
    return read_pydantic_yaml(TypeAdapter(SpecificRunSpec))(filename)


def read_load_test_spec(filename: str) -> LoadTestSpec:
    return read_pydantic_yaml(LoadTestSpec)(filename)


def read_observation_spec(filename: str) -> ObservationSpec:
    return read_pydantic_yaml(ObservationSpec)(filename)
