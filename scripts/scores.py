import math
from typing import Literal

def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")