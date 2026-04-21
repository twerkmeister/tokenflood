from typing import Literal
from pydantic import NonNegativeFloat, Field
from tokenflood.constants import OBSERVATION_SPEC_FILE

from tokenflood.models.load_types.load_type import SpecificLoadType
from tokenflood.models.run_specs.run_spec import RunSpec
from tokenflood.models.validation_types import PositiveFloat


class ObservationSpec(RunSpec, frozen=True):
    type: Literal["observation"] = "observation"
    name: str
    duration_hours: PositiveFloat
    polling_interval_minutes: PositiveFloat
    load_type: SpecificLoadType
    num_requests: int = Field(ge=1)
    within_seconds: NonNegativeFloat

    def num_polls(self) -> int:
        return int((self.duration_hours * 60) / self.polling_interval_minutes)

    def total_num_requests(self) -> int:
        return self.num_requests * self.num_polls()

    def requests_per_second_during_polling(self) -> float:
        return self.num_requests / self.within_seconds

    def get_inter_polling_pause(self) -> float:
        return self.polling_interval_minutes * 60 - self.within_seconds

    @property
    def run_spec_file(self) -> str:
        return OBSERVATION_SPEC_FILE