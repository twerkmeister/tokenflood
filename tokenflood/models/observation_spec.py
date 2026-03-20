from typing import Tuple

from pydantic import BaseModel, NonNegativeFloat, Field

from tokenflood.models.budget import Budget
from tokenflood.models.heuristic_task import HeuristicTask
from tokenflood.models.load_type import LoadType
from tokenflood.models.token_cost_aware import TokenCostAware
from tokenflood.models.token_set import TokenSet


class ObservationSpec(BaseModel, TokenCostAware, frozen=True):
    name: str
    duration_hours: NonNegativeFloat
    polling_interval_minutes: NonNegativeFloat
    load_type: LoadType
    num_requests: int = Field(ge=1)
    within_seconds: NonNegativeFloat
    task: HeuristicTask
    token_set: TokenSet
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

    def requests_per_second_during_polling(self) -> float:
        return self.num_requests / self.within_seconds

    def get_inter_polling_pause(self) -> float:
        return self.polling_interval_minutes * 60 - self.within_seconds
