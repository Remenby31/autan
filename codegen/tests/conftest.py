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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="cpu: host library, GPU tests deselected; gpu: AUTAN_SPIKE_LIB must be libautan_spike_gpu",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--backend") == "gpu":
        if "AUTAN_SPIKE_LIB" not in os.environ:
            raise pytest.UsageError("--backend gpu needs AUTAN_SPIKE_LIB=.../libautan_spike_gpu.so")
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
