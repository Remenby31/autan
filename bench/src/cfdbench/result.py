import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

from cfdbench.errors import BenchError, Code

SCHEMA_ID = "cfd-bench-result/1"


@cache
def schema() -> dict[str, Any]:
    return json.loads(resources.files("cfdbench").joinpath("result.schema.json").read_text())


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
    names = [b["name"] for b in doc["benchmarks"]]
    if len(names) != len(set(names)):
        raise BenchError(Code.SCHEMA_INVALID, "benchmark names are not unique")


def load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchError(Code.RESULT_UNREADABLE, f"{path}: {exc}") from exc
    validate(doc)
    return doc


def write(doc: dict[str, Any], path: Path) -> None:
    validate(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def peak_bandwidth(doc: dict[str, Any]) -> float:
    """Highest median bandwidth (GB/s) among the STREAM benchmarks of a result."""
    medians = [b["stats"]["median"] for b in doc["benchmarks"] if b["name"].startswith("stream.")]
    if not medians:
        raise BenchError(Code.NO_STREAM, "the result holds no stream.* benchmark")
    return max(medians)
