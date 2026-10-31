import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autan_ui.registry_io import read_history

TREND_POINTS = 48


@dataclass
class _Trend:
    offset: int = 0
    first: dict[str, float] | None = None
    iteration: int | None = None
    orders: list[float] = field(default_factory=list)


def residual_orders(first: dict[str, float], residuals: dict[str, float]) -> float | None:
    """Orders of magnitude the slowest equation has dropped since iteration 1: the quantity
    the steady convergence criterion (`time.residual_orders`, case-schema.md) compares."""
    drops = []
    for name, value in residuals.items():
        start = first.get(name)
        if isinstance(start, float | int) and _positive(start) and _positive(value):
            drops.append(math.log10(start / value))
    return min(drops) if drops else None


def _positive(value: object) -> bool:
    return isinstance(value, float | int) and not isinstance(value, bool) and 0 < value < math.inf


def downsample(values: list[float], points: int = TREND_POINTS) -> list[float]:
    if len(values) <= points:
        return [round(v, 3) for v in values]
    step = (len(values) - 1) / (points - 1)
    return [round(values[round(k * step)], 3) for k in range(points)]


class Summaries:
    """Per-run iteration, residual drop and its trend, read incrementally from history.jsonl."""

    def __init__(self) -> None:
        self._trends: dict[str, _Trend] = {}

    def update(self, directory: Path, *, live: bool) -> dict[str, Any] | None:
        trend = self._trends.setdefault(directory.name, _Trend())
        try:
            size = (directory / "history.jsonl").stat().st_size
        except OSError:
            size = 0
        if size < trend.offset:
            trend = self._trends[directory.name] = _Trend()
        try:
            chunk = read_history(directory, trend.offset, live=live)
        except ValueError:
            return None
        for record in chunk.records:
            iteration = record.get("iteration") if isinstance(record, dict) else None
            if type(iteration) is not int:
                continue
            residuals = record.get("residuals")
            if not isinstance(residuals, dict):
                residuals = {}
            if trend.first is None:
                trend.first = residuals
            trend.iteration = iteration
            orders = residual_orders(trend.first, residuals)
            if orders is not None:
                trend.orders.append(orders)
        trend.offset = chunk.offset
        if trend.iteration is None:
            return None
        return {
            "iteration": trend.iteration,
            "orders": round(trend.orders[-1], 3) if trend.orders else None,
            "trend": downsample(trend.orders),
        }

    def forget(self, names: set[str]) -> None:
        for name in names:
            self._trends.pop(name, None)
