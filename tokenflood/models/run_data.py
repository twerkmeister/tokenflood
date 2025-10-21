from typing import List

from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from tokenflood.models.results import Results
from tokenflood.models.run_spec import RunSpec


class RunData(BaseModel, frozen=True):
    run_spec: RunSpec
    responses: List[ModelResponse]
    results: Results
