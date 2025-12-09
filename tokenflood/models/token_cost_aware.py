from typing import Tuple


class TokenCostAware:
    def get_input_output_token_cost(self) -> Tuple[int, int]:
        raise NotImplementedError
