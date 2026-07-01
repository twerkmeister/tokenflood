from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd

from tokenflood.models.util import numeric


@dataclass
class AggregationFunc:
    f: Callable[[pd.Series], numeric | str | datetime]
    name: str
    order: float
    field: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, AggregationFunc):
            return self.name == other.name and self.field == other.field
        return False
