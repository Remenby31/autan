import json

from conftest import bench, make_result

from cfdbench.cli import main

QUIET = [200.0, 201.0, 199.0, 200.5, 199.5]
NOISY = [180.0, 220.0, 190.0, 210.0, 200.0]


def test_baseline_keeps_only_clean_low_noise_results(write_json, tmp_path, capsys):
    out = tmp_path / "gb10.json"
    good = make_result([bench("stream.copy", QUIET), bench("stream.triad", [x + 10 for x in QUIET])])
    assert main(["baseline", str(write_json("good.json", good)), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == good
    capsys.readouterr()
    assert main(["peak", str(out)]) == 0
    assert float(capsys.readouterr().out) == 210.0

    noisy = make_result([bench("stream.copy", QUIET), bench("stream.triad", NOISY)])
    assert main(["baseline", str(write_json("noisy.json", noisy)), "--out", str(out)]) == 2
    assert "error[CV_TOO_HIGH]" in capsys.readouterr().err

    dirty = make_result([bench("stream.copy", QUIET)], dirty=True)
    assert main(["baseline", str(write_json("dirty.json", dirty)), "--out", str(out)]) == 2
    assert "error[DIRTY_TREE]" in capsys.readouterr().err
    assert json.loads(out.read_text()) == good


def test_validate_rejects_documents_outside_the_schema(write_json, capsys):
    doc = make_result([bench("stream.copy", QUIET)])
    assert main(["validate", str(write_json("ok.json", doc))]) == 0
    doc["conditions"]["idle_check"]["compute_apps_max"] = 1
    assert main(["validate", str(write_json("busy.json", doc))]) == 2
    assert "error[SCHEMA_INVALID]" in capsys.readouterr().err
