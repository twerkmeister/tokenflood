from typing import Annotated

from pydantic import Field

from tokenflood.models.run_specs.load_test_spec import LoadTestSpec
from tokenflood.models.run_specs.observation_spec import ObservationSpec

SpecificRunSpec = Annotated[LoadTestSpec | ObservationSpec, Field(discriminator="type")]
