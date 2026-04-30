from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioSpec:
    long_k: int = 20
    short_k: int = 20
    long_short: bool = False
    max_weight: float = 0.1  # 10% cap


def equal_weight_weights(symbols: list[str], max_weight: float) -> dict[str, float]:
    if len(symbols) == 0:
        return {}
    w = 1.0 / len(symbols)
    w = float(min(w, max_weight))
    # if capped, remaining cash stays uninvested (conservative)
    return {s: w for s in symbols}


def make_long_short_weights(
    longs: list[str],
    shorts: list[str],
    max_weight: float,
) -> dict[str, float]:
    wl = equal_weight_weights(longs, max_weight=max_weight)
    ws = equal_weight_weights(shorts, max_weight=max_weight)
    # scale shorts to -1 exposure on their side (dollar neutral-ish if both sides fill)
    if len(ws) > 0:
        for k in list(ws.keys()):
            ws[k] = -ws[k]
    w = wl | ws
    # normalize gross exposure to 1 if needed
    gross = sum(abs(x) for x in w.values())
    if gross > 1.0 and gross > 0:
        w = {k: float(v / gross) for k, v in w.items()}
    return w


def topk_from_scores(scores: dict[str, float], k: int, ascending: bool = False) -> list[str]:
    if k <= 0:
        return []
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=not ascending)
    return [s for s, _ in items[:k]]

