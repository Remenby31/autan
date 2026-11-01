import json

import pytest
from autan_bench.cli import main
from conftest import bench, make_result

QUIET = [200.0, 201.0, 199.0, 200.5, 199.5]
NOISY = [180.0, 220.0, 190.0, 210.0, 200.0]
# GB10 L2 is 24 MiB; an array of 2^20 FP64 values (8 MiB) fits in it.
IN_CACHE = {"elements": 1 << 20, "element_bytes": 8}


def test_baseline_promotes_a_clean_low_noise_out_of_cache_result(write_json, tmp_path, capsys):
    out = tmp_path / "gb10.json"
    good = make_result(
        [bench("stream.copy", QUIET), bench("stream.triad", [x + 10 for x in QUIET])]
    )
    assert main(["baseline", str(write_json("good.json", good)), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == good
    capsys.readouterr()
    assert main(["peak", str(out)]) == 0
    assert float(capsys.readouterr().out) == 210.0


def test_baseline_keeps_only_the_selected_benchmarks(write_json, tmp_path, capsys):
    out = tmp_path / "tracked.json"
    doc = make_result([bench("stream.copy", QUIET), bench("stream.triad", NOISY)])
    path = str(write_json("doc.json", doc))
    assert main(["baseline", path, "--out", str(out), "--select", "stream.co*"]) == 0
    assert [b["name"] for b in json.loads(out.read_text())["benchmarks"]] == ["stream.copy"]
    out.unlink()
    assert main(["baseline", path, "--out", str(out), "--select", "stream.nothing*"]) == 2
    assert "error[MISSING_BENCHMARK]" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("doc", "code"),
    [
        (make_result([bench("stream.copy", QUIET), bench("stream.triad", NOISY)]), "CV_TOO_HIGH"),
        (make_result([bench("stream.copy", QUIET)], dirty=True), "DIRTY_TREE"),
        (make_result([bench("stream.copy", QUIET)], image=None), "IMAGE_UNKNOWN"),
        (make_result([bench("stream.copy", QUIET)], config=IN_CACHE), "FOOTPRINT_TOO_SMALL"),
    ],
)
def test_baseline_refuses_untrustworthy_results(write_json, tmp_path, capsys, doc, code):
    out = tmp_path / "gb10.json"
    assert main(["baseline", str(write_json("doc.json", doc)), "--out", str(out)]) == 2
    assert f"error[{code}]" in capsys.readouterr().err
    assert not out.exists()


def test_validate_rejects_documents_outside_the_schema(write_json, tmp_path, capsys):
    doc = make_result([bench("stream.copy", QUIET)])
    assert main(["validate", str(write_json("ok.json", doc))]) == 0

    busy = json.loads(json.dumps(doc))
    busy["conditions"]["idle_check"]["compute_apps_max"] = 1
    assert main(["validate", str(write_json("busy.json", busy))]) == 2
    assert "error[SCHEMA_INVALID]" in capsys.readouterr().err

    stale = json.loads(json.dumps(doc))
    stale["benchmarks"][0]["stats"]["median"] = 100.0
    assert main(["validate", str(write_json("stale.json", stale))]) == 2
    assert "error[SCHEMA_INVALID]" in capsys.readouterr().err

    nan = tmp_path / "nan.json"
    nan.write_text(json.dumps(doc).replace('"cv": ', '"cv": NaN, "x": ', 1))
    assert main(["validate", str(nan)]) == 2
    assert "error[RESULT_UNREADABLE]" in capsys.readouterr().err


METAL_GPU = {
    "name": "Apple M4 Pro",
    "registry_id": "4294968922",
    "gpu_cores": 20,
    "recommended_max_working_set_bytes": 36 * 2**30,
    "memory_bytes": 48 * 2**30,
}


def as_metal(doc: dict) -> dict:
    doc["provenance"].update(backend="metal", precision="fp32", gpu=METAL_GPU)
    doc["provenance"]["devices"]["kind"] = "metal"
    doc["conditions"]["idle_check"]["compute_apps_max"] = None
    doc["conditions"]["during_run"].update(compute_apps_max=None, clocks_event_reasons=None)
    return doc


def set_path(doc: dict, path: str, value: object) -> dict:
    *parents, last = path.split(".")
    node = doc
    for key in parents:
        node = node[key]
    node[last] = value
    return doc


@pytest.mark.parametrize(
    ("metal", "path", "value", "valid"),
    [
        (False, "provenance.backend", "cuda", True),
        (True, "provenance.backend", "metal", True),
        (True, "provenance.precision", "fp64", False),
        (False, "conditions.idle_check.compute_apps_max", None, False),
        (False, "conditions.during_run.compute_apps_max", None, False),
        (False, "conditions.during_run.clocks_event_reasons", None, False),
        (False, "provenance.gpu", METAL_GPU, False),
        (
            True,
            "provenance.gpu",
            make_result([bench("stream.copy", QUIET)])["provenance"]["gpu"],
            False,
        ),
        (False, "provenance.devices.kind", "metal", False),
    ],
)
def test_validate_binds_backend_device_and_telemetry(write_json, metal, path, value, valid):
    doc = make_result([bench("stream.copy", QUIET)])
    doc = set_path(as_metal(doc) if metal else doc, path, value)
    assert main(["validate", str(write_json("doc.json", doc))]) == (0 if valid else 2)
