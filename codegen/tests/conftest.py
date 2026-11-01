"""Builds the spike library once per session (CMake, host compiler) unless AUTAN_SPIKE_LIB is set."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PRECISIONS = ["fp64", "fp32+acc64", "fp32"]
KINDS = ["tet", "hex", "poly", "hybrid"]


def eps(policy: str) -> float:
    """Machine epsilon of the compute type of a policy."""
    return float(np.finfo(np.float64 if policy == "fp64" else np.float32).eps)


# Policy and strategies a device backend refuses, with the code it refuses them with: Metal has
# no FP64, and no atomics on the double-float accumulation type of fp32+acc64.
REFUSED = {
    "metal": {
        ("fp64", None): "E1013",
        ("fp32+acc64", "atomic"): "E1108",
        ("fp32+acc64", "warp-aggregated"): "E1108",
    }
}


def refused(backend: str, policy: str, strategy: str | None = None) -> str | None:
    """The error code `backend` refuses `policy` (and `strategy`) with, or None."""
    table = REFUSED.get(backend, {})
    return table.get((policy, None)) or table.get((policy, strategy))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend",
        choices=["cpu", "gpu", "metal"],
        default="cpu",
        help="cpu: host library, device tests deselected; gpu or metal: AUTAN_SPIKE_LIB must be "
        "libautan_spike_gpu or libautan_spike_metal",
    )


@pytest.fixture(scope="session")
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--backend"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--backend") in ("gpu", "metal"):
        if "AUTAN_SPIKE_LIB" not in os.environ:
            msg = "--backend gpu|metal needs AUTAN_SPIKE_LIB=.../libautan_spike_{gpu,metal}"
            raise pytest.UsageError(msg)
        return
    gpu = [i for i in items if i.get_closest_marker("gpu")]
    if gpu:
        config.hook.pytest_deselected(items=gpu)
        items[:] = [i for i in items if not i.get_closest_marker("gpu")]


@pytest.fixture(scope="session", autouse=True)
def spike_library(tmp_path_factory: pytest.TempPathFactory) -> str:
    if "AUTAN_SPIKE_LIB" not in os.environ:
        build = tmp_path_factory.mktemp("build")
        subprocess.run(
            ["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DAUTAN_BUILD_TESTS=OFF", "-DAUTAN_WARNINGS_AS_ERRORS=ON", f"-DPython3_EXECUTABLE={sys.executable}"],
            check=True, capture_output=True,
        )  # fmt: skip
        subprocess.run(
            ["cmake", "--build", str(build), "--target", "autan_spike"],
            check=True,
            capture_output=True,
        )
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        os.environ["AUTAN_SPIKE_LIB"] = str(build / "codegen" / f"libautan_spike{suffix}")
    return os.environ["AUTAN_SPIKE_LIB"]
