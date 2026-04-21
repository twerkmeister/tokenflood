from typing import Annotated

from pydantic import Field

from tokenflood.models.run_specs.load_spec import LoadSpec
from tokenflood.models.run_specs.observation_spec import ObservationSpec

SpecificRunSpec = Annotated[LoadSpec | ObservationSpec, Field(discriminator="type")]