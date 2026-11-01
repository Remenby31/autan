import json
import math
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

from autan_bench.errors import BenchError, Code
from autan_bench.stats import summarize

SCHEMA_ID = "autan.bench-result/1"


@cache
def schema() -> dict[str, Any]:
    return json.loads(resources.files("autan_bench").joinpath("result.schema.json").read_text())


def validate(doc: dict[str, Any]) -> None:
    validator = jsonschema.Draft202012Validator(
        schema(), format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
    )
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        where = "/".join(str(p) for p in errors[0].absolute_path) or "<root>"
        raise BenchError(
            Code.SCHEMA_INVALID, f"{where}: {errors[0].message} ({len(errors)} error(s))"
        )
    provenance = doc["provenance"]
    if provenance["devices"]["kind"] != provenance["backend"]:
        raise BenchError(
            Code.SCHEMA_INVALID,
            f"devices.kind {provenance['devices']['kind']!r} != backend {provenance['backend']!r}",
        )
    names = [b["name"] for b in doc["benchmarks"]]
    if len(names) != len(set(names)):
        raise BenchError(Code.SCHEMA_INVALID, "benchmark names are not unique")
    for b in doc["benchmarks"]:
        expected = summarize(b["samples"])
        for key, value in expected.items():
            if not math.isclose(b["stats"][key], value, rel_tol=1e-12, abs_tol=1e-12):
                raise BenchError(
                    Code.SCHEMA_INVALID,
                    f"{b['name']}: stats.{key} is {b['stats'][key]!r}, its samples give {value!r}",
                )


def _reject_constant(name: str) -> float:
    msg = f"{name} is not a finite number"
    raise ValueError(msg)


def load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise BenchError(Code.RESULT_UNREADABLE, f"{path}: {exc}") from exc
    validate(doc)
    return doc


def write(doc: dict[str, Any], path: Path) -> None:
    validate(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def peak_bandwidth(doc: dict[str, Any]) -> float:
    """Highest median bandwidth (GB/s) among the STREAM benchmarks of a result."""
    medians = [b["stats"]["median"] for b in doc["benchmarks"] if b["name"].startswith("stream.")]
    if not medians:
        raise BenchError(Code.NO_STREAM, "the result holds no stream.* benchmark")
    return max(medians)


_SAME_IN_POOLED_RUNS = (
    "solver",
    "binary_sha256",
    "image",
    "backend",
    "precision",
    "devices",
    "gpu",
)


def pool(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge runs of one binary, image, GPU and configuration into one result.

    Each benchmark gets the samples of every run, in run order, and statistics recomputed from
    them, so its median spans the drift between runs. ``pooled_from`` lists the runs' creation
    times; ``conditions`` are those of the last run.
    """
    first = docs[0]
    for doc in docs[1:]:
        for key in _SAME_IN_POOLED_RUNS:
            if doc["provenance"][key] != first["provenance"][key]:
                raise BenchError(Code.INCOMPATIBLE, f"runs differ in provenance.{key}")
        if doc["suite"] != first["suite"] or doc["config"] != first["config"]:
            raise BenchError(Code.INCOMPATIBLE, "runs differ in suite or config")
        if [b["name"] for b in doc["benchmarks"]] != [b["name"] for b in first["benchmarks"]]:
            raise BenchError(Code.INCOMPATIBLE, "runs hold different benchmarks")
    created = [d["created"] for d in docs]
    if len(set(created)) != len(created):
        raise BenchError(Code.INCOMPATIBLE, "the same run is given twice")
    pooled = json.loads(json.dumps(docs[-1]))
    for i, bench in enumerate(pooled["benchmarks"]):
        bench["samples"] = [x for d in docs for x in d["benchmarks"][i]["samples"]]
        bench["stats"] = summarize(bench["samples"])
    pooled["pooled_from"] = created
    validate(pooled)
    return pooled
