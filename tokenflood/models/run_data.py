from typing import List

import pandas as pd
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from tokenflood.models.results import Results
from tokenflood.models.run_spec import RunSpec


class RunData(BaseModel, frozen=True):
    run_spec: RunSpec
    responses: List[ModelResponse]
    results: Results

    def as_dataframe(self) -> pd.DataFrame:
        """Create a dataframe with the requests per second phase and the result data."""
        requests_per_second_column = [
            self.run_spec.requests_per_second
        ] * self.run_spec.total_num_requests
        run_spec_df = pd.DataFrame(
            {"requests_per_second_at_the_time": requests_per_second_column}
        )
        return pd.concat([run_spec_df, self.results.as_dataframe()], axis=1)
