"""The `autan-vv` command line, through its outputs."""

import json
from pathlib import Path

import pytest
from autan_vv.cli import main


def test_order_command_reports_the_order_of_synthetic_errors(tmp_path: Path) -> None:
    levels = [
        {"n_cells": n**3, "volume": 1.0, "errors": {"L2": 0.5 * n**-2.0}} for n in (8, 16, 32)
    ]
    src = tmp_path / "errors.json"
    src.write_text(json.dumps({"label": "synthetic", "dim": 3, "levels": levels}))
    assert (
        main(
            [
                "order",
                str(src),
                "--out",
                str(tmp_path / "r.json"),
                "--plot",
                str(tmp_path / "r.png"),
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["fits"]["L2"]["order"] == pytest.approx(2.0, abs=1e-12)
    assert report["levels"][0]["h"] == pytest.approx(1 / 8)
    assert (tmp_path / "r.png").stat().st_size > 0


def test_order_command_returns_2_on_a_coded_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "errors.json"
    src.write_text(
        json.dumps({"dim": 3, "levels": [{"n_cells": 8, "volume": 1.0, "errors": {"L2": 1.0}}]})
    )
    assert main(["order", str(src), "--out", str(tmp_path / "r.json")]) == 2
    assert "V0501" in capsys.readouterr().err


def test_mms_command_writes_header_and_json(tmp_path: Path) -> None:
    assert main(["mms", "euler", "--out", str(tmp_path)]) == 0
    doc = json.loads((tmp_path / "euler.json").read_text())
    assert doc["schema"] == "autan.mms/1"
    assert "namespace autan::mms::euler" in (tmp_path / "euler.hpp").read_text()


def test_cases_command_lists_the_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cases", "list"]) == 0
    assert "tmr-2dzp-flatplate" in capsys.readouterr().out
    assert main(["cases", "fetch", "no-such-case"]) == 2
    assert "V0604" in capsys.readouterr().err
