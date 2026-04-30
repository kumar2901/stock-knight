from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """
    Simple cost model (placeholder) to keep the pipeline realistic.

    Upgrade path:
    - instrument-specific costs (cash vs futures)
    - slippage as function of volatility and %ADV
    - borrow constraints for cash shorts (research mode only)
    """

    bps_per_side: float = 10.0  # 10 bps each buy/sell side (rough)

    def cost_fraction(self) -> float:
        return self.bps_per_side / 10_000.0


def estimate_turnover(prev_w: dict[str, float], new_w: dict[str, float]) -> float:
    keys = set(prev_w) | set(new_w)
    return sum(abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys) / 2.0

