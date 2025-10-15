from typing import List

from pydantic import BaseModel

class HeuristicTestSpec(BaseModel):
    name: str
    requests_per_second: float
    prompt_lengths: List[int]
    prefix_lengths: List[int]
    output_lengths: List[int]
    test_length_in_seconds: int
