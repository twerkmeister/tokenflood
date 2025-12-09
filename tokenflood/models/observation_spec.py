from typing import Tuple

from pydantic import BaseModel, NonNegativeFloat

from tokenflood.models.budget import Budget
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import LoadType
from tokenflood.models.token_cost_aware import TokenCostAware
from tokenflood.models.token_set import TokenSet
from tokenflood.models.validation_types import (
    NonNegativeInteger,
    PositiveUniqueIntegers,
)


class ObservationSpec(BaseModel, TokenCostAware, frozen=True):
    name: str
    duration_hours: NonNegativeFloat
    polling_interval_minutes: NonNegativeFloat
    load_type: LoadType
    num_requests: NonNegativeInteger
    within_seconds: NonNegativeFloat
    task: HeuristicTask
    token_set: TokenSet
    percentiles: PositiveUniqueIntegers
    budget: Budget = Budget()

    def num_polls(self) -> int:
        return int((self.duration_hours * 60) / self.polling_interval_minutes)

    def total_num_requests(self) -> int:
        return self.num_requests * self.num_polls()

    def get_input_output_token_cost(self) -> Tuple[int, int]:
        return (
            self.total_num_requests() * self.load_type.prompt_length,
            self.total_num_requests() * self.load_type.output_length,
        )
