from pydantic import BaseModel

from tokenflood.models.endpoint_spec import EndpointSpec
from tokenflood.models.load_types.load_type import SpecificLoadType
from tokenflood.util import get_date_str, get_run_name


class RunSpec(BaseModel, frozen=True):
    type: str
    name: str
    load_type: SpecificLoadType

    def get_run_name(self, endpoint_spec: EndpointSpec) -> str:
        date_str = get_date_str()
        return get_run_name(date_str, self.type, self.name, endpoint_spec)

    @property
    def run_spec_file(self) -> str:
        raise NotImplementedError