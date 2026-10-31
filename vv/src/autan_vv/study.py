"""Observed-order studies: errors on a mesh family to a JSON report and a log-log plot."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from autan_vv.errors import VVError
from autan_vv.order import fit_order, mesh_size

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

REPORT_SCHEMA = "autan.order-study/1"


def order_study(label: str, dim: int, levels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fit the observed order of every error norm over a family.

    Each level is {"n_cells": int, "volume": float, "errors": {norm: value}} plus any provenance
    keys (mesh hash, file), which are copied to the report unchanged.
    """
    if len(levels) < 2:
        msg = "an order study needs at least two levels"
        raise VVError(msg, code="V0501")
    norms = list(levels[0]["errors"])
    rows = [{**lv, "h": mesh_size(int(lv["n_cells"]), float(lv["volume"]), dim)} for lv in levels]
    fits = {}
    for norm in norms:
        fit = fit_order([r["h"] for r in rows], [r["errors"][norm] for r in rows])
        fits[norm] = {k: _json_ready(v) for k, v in asdict(fit).items()}
    return {"schema": REPORT_SCHEMA, "label": label, "dim": dim, "levels": rows, "fits": fits}


def _json_ready(value: Any) -> Any:
    """NaN (an undefined statistic, e.g. a CI from two levels) becomes null."""
    if isinstance(value, tuple | list):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(report: dict[str, Any], json_path: Path, plot_path: Path | None = None) -> None:
    """Write the report as JSON and, optionally, its log-log error plot."""
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if plot_path is None:
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")
    h = [r["h"] for r in report["levels"]]
    for norm, fit in report["fits"].items():
        e = [r["errors"][norm] for r in report["levels"]]
        (line,) = ax.loglog(h, e, "o", label=f"{norm}: p = {fit['order']:.2f}")
        ax.loglog(
            h, [fit["constant"] * hi ** fit["order"] for hi in h], "-", color=line.get_color(), lw=1
        )
    ax.set_xlabel("h = (V/N)^(1/d)")
    ax.set_ylabel("error")
    ax.set_title(report["label"], fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(visible=True, which="both", alpha=0.3)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
