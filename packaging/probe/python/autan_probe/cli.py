"""``autan-probe``: report what the installed wheel can run on this machine.

Prints one line per check, or JSON with ``--json``. Exit status 0 when nothing failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Any CUDA 13.x runtime runs on a driver that supports CUDA 13.0, i.e. driver 580 or later (CUDA
# minor version compatibility). cudaDriverGetVersion reports the highest version it supports.
CUDA_RUNTIME_MIN = 13000


def _check(name: str, status: str, detail: str, fix: str | None = None) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "fix": fix}


def _cuda_version(encoded: int) -> str:
    return f"{encoded // 1000}.{encoded % 1000 // 10}"


def _has_sass(architectures: str, capability: str) -> bool:
    """Whether the fatbin holds a cubin this GPU runs: same major, minor not above the GPU's."""
    major, minor = (int(part) for part in capability.split("."))
    for arch in architectures.split(","):
        number, _, kind = arch.partition("-")
        if kind == "real" and int(number) // 10 == major and int(number) % 10 <= minor:
            return True
    return False


def _gpu_checks(cuda: dict[str, Any]) -> list[dict[str, Any]]:
    if not cuda["compiled"]:
        if sys.platform == "darwin":
            return [_check("gpu", "info", "Apple Silicon: CPU backend (Metal comes in M1b)")]
        return [_check("gpu", "info", "this wheel has no GPU code")]
    if Path("/dev/kfd").exists() and not cuda["devices"]:
        return [
            _check(
                "gpu",
                "warn",
                "AMD GPU found (/dev/kfd); this wheel carries CUDA code only",
                "ROCm wheels are not published yet",
            )
        ]
    if cuda["driver_version"] == 0:
        return [
            _check(
                "gpu",
                "info",
                "no NVIDIA driver (libcuda.so.1 not found): CPU backend",
                "install the NVIDIA driver (>= 580) to use the GPU; nothing else is needed",
            )
        ]
    if cuda["driver_version"] < CUDA_RUNTIME_MIN:
        return [
            _check(
                "driver",
                "fail",
                f"driver supports CUDA {_cuda_version(cuda['driver_version'])}, "
                f"the wheel needs CUDA {_cuda_version(CUDA_RUNTIME_MIN)}",
                "upgrade the NVIDIA driver to 580 or later",
            )
        ]
    checks = [
        _check(
            "driver",
            "ok",
            f"supports CUDA {_cuda_version(cuda['driver_version'])}; "
            f"runtime {_cuda_version(cuda['runtime_version'])}",
        )
    ]
    if cuda["error"]:
        checks.append(_check("gpu", "fail", cuda["error"], "run nvidia-smi to check the driver"))
    for device in cuda["devices"]:
        label = f"gpu{device['index']}"
        name = f"{device['name']} (sm_{device['compute_capability'].replace('.', '')})"
        if device["error"]:
            checks.append(
                _check(label, "fail", f"{name}: {device['error']}", "report it with this output")
            )
        elif not _has_sass(cuda["architectures"], device["compute_capability"]):
            checks.append(
                _check(
                    label,
                    "warn",
                    f"{name}: no native code in this wheel, kernels JIT-compiled from PTX",
                    "slower first launch; a newer release will carry SASS for this GPU",
                )
            )
        else:
            checks.append(
                _check(label, "ok", f"{name}, {device['memory_gb']:.0f} GB, native code, axpy ok")
            )
    return checks


def _path_check() -> dict[str, Any]:
    command = Path(sys.argv[0]).name
    found = shutil.which(command)
    if found:
        return _check("path", "ok", f"{command} -> {found}")
    directory = Path(sys.argv[0]).absolute().parent
    return _check(
        "path",
        "warn",
        f"{command} is not on PATH ({directory})",
        f'export PATH="{directory}:$PATH", or run `uv tool update-shell`',
    )


def report() -> dict[str, Any]:
    """Every check, as the JSON document ``autan-probe --json`` prints."""
    checks = [
        _check(
            "python",
            "ok",
            f"{platform.python_version()} {platform.machine()} ({sys.executable})",
        )
    ]
    try:
        from autan_probe import _native  # noqa: PLC0415 - a broken wheel must still report
    except ImportError as error:
        checks.append(
            _check("native", "fail", str(error), "reinstall: uv tool install --reinstall autan")
        )
        cuda: dict[str, Any] | None = None
    else:
        checks.append(_check("native", "ok", f"compiled for {_native.cpu_target()}"))
        cuda = _native.cuda_report()
        checks += _gpu_checks(cuda)
    checks.append(_path_check())
    return {
        "probe": version("autan-probe"),
        "platform": f"{sys.platform}-{platform.machine()}",
        "ok": all(c["status"] != "fail" for c in checks),
        "checks": checks,
        "cuda": cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="autan-probe", description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail unless every GPU runs native code (for GPU CI runners)",
    )
    args = parser.parse_args()
    document = report()
    if args.require_gpu:
        gpus = [c for c in document["checks"] if c["name"].startswith("gpu")]
        if not gpus or any(c["status"] != "ok" for c in gpus):
            document["ok"] = False
    if args.json:
        sys.stdout.write(json.dumps(document, indent=2) + "\n")
    else:
        for c in document["checks"]:
            line = f"{c['name']:<7} {c['status']:<5} {c['detail']}"
            if c["fix"]:
                line += f"\n{'':<13} fix: {c['fix']}"
            sys.stdout.write(line + "\n")
    return 0 if document["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
