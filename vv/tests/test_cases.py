"""Reference case library: pinned downloads, checksums and parsers.

Offline tests use local `file://` URLs and format snippets written from the format descriptions.
Network tests (`-m network`) fetch every case and cross-check the values transcribed in the
catalogue from the TMR pages against the downloaded data files, a second source.
"""

import hashlib
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
from autan_vv.cases import (
    Case,
    CaseFile,
    Table,
    fetch,
    get_case,
    list_cases,
    parse,
    pointers,
    read_columns,
    read_plot3d_2d,
    read_tape_1981,
    read_tecplot,
)
from autan_vv.errors import VVError


def _case(files: tuple[CaseFile, ...]) -> Case:
    return Case("local", "local", {}, {}, (), "", files)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fetch_verifies_files_and_archive_members(tmp_path: Path) -> None:
    payload = b'variables="x","y"\nzone t="a"\n1 2\n3 4\n'
    src = tmp_path / "src.dat"
    src.write_bytes(payload)
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("deep/dir/member.dat", payload)
    good = (
        CaseFile(
            "plain.dat", "reference", src.as_uri(), _sha(payload), len(payload), "tecplot", ""
        ),
        CaseFile(
            "member.dat",
            "grid",
            archive.as_uri(),
            _sha(archive.read_bytes()),
            archive.stat().st_size,
            "tecplot",
            "",
            member="deep/dir/member.dat",
            member_sha256=_sha(payload),
        ),
    )
    paths = fetch(_case(good), tmp_path / "cache")
    assert [p.read_bytes() for p in paths] == [payload, payload]
    assert parse(good[0], paths[0])[0].column("y").tolist() == [2.0, 4.0]

    bad_file = CaseFile("x.dat", "reference", src.as_uri(), "0" * 64, len(payload), "tecplot", "")
    with pytest.raises(VVError) as err:
        fetch(_case((bad_file,)), tmp_path / "cache2")
    assert err.value.code == "V0601"
    bad_member = CaseFile(
        "m.dat",
        "grid",
        archive.as_uri(),
        _sha(archive.read_bytes()),
        archive.stat().st_size,
        "tecplot",
        "",
        member="deep/dir/member.dat",
        member_sha256="0" * 64,
    )
    with pytest.raises(VVError) as err:
        fetch(_case((bad_member,)), tmp_path / "cache3")
    assert err.value.code == "V0602"


def test_tecplot_reader_handles_tmr_header_variants(tmp_path: Path) -> None:
    path = tmp_path / "t.dat"
    path.write_text(
        "# comment\n"
        'VARIABLES = "alpha, deg" "cl"\n'
        'zone, t="first"\n 0.0 1.0D-01\n 1.0, 2.0E+00\n'
        'ZONE T="second", I=  2, F=POINT\n 3.0\n 4.0 5.0 6.0\n'
    )
    tables = read_tecplot(path)
    assert [t.title for t in tables] == ["first", "second"]
    assert tables[0].variables == ("alpha, deg", "cl")
    np.testing.assert_array_equal(tables[0].data, [[0.0, 0.1], [1.0, 2.0]])
    np.testing.assert_array_equal(tables[1].column("cl"), [4.0, 6.0])
    path.write_text('variables="a","b"\nzone t="z"\n1 2 3\n')
    with pytest.raises(VVError) as err:
        read_tecplot(path)
    assert err.value.code == "V0603"


def test_columns_reader_skips_headers(tmp_path: Path) -> None:
    path = tmp_path / "c.txt"
    path.write_text("RAE 2822\n  x/c  z/c\n\n0.0 0.0 0.0\n0.5 -0.01 0.02\n")
    np.testing.assert_array_equal(read_columns(path).data, [[0.0, 0.0, 0.0], [0.5, -0.01, 0.02]])


def test_plot3d_reader_orders_i_fastest(tmp_path: Path) -> None:
    path = tmp_path / "g.p2dfmt"
    path.write_text("1\n3 2\n0 1 2 0 1 2\n0 0 0 1 1 1\n")
    ((x, y),) = read_plot3d_2d(path)
    np.testing.assert_array_equal(x, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(y, [[0, 0, 0], [1, 1, 1]])


def test_tape_decoder_applies_the_min_max_scaling(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_text(
        "----------------- FILE NUMBER     7-----------------\n"
        " 0.100000E 02 0.200000E-01\n"
        " 0.000000E 00-0.200000E-01\n"
        "     0 10000\n"
        " 10000 20000\n"
        "-------------- END OF FILE NUMBER     7-------------\n"
    )
    values = read_tape_1981(path)[7]
    np.testing.assert_allclose(values[0], [0.0, 0.02])
    assert values[1, 0] == 10.0
    assert math.isnan(values[1, 1])


def test_catalogue_is_complete() -> None:
    names = {c.name for c in list_cases()}
    assert {
        "tmr-2dzp-flatplate",
        "tmr-2dbc-bump",
        "tmr-2dn00-naca0012",
        "onera-m6",
        "rae2822",
    } <= names
    for case in list_cases():
        assert case.references
        assert case.licence
        for f in case.files:
            assert f.url.startswith("https://")
            assert len(f.sha256) == 64
            assert (f.member is None) == (f.member_sha256 is None)
    assert {"HLPW5", "DPW7", "DPW8"} <= set(pointers())
    with pytest.raises(VVError) as err:
        get_case("nope")
    assert err.value.code == "V0604"


# --- network: every case downloads, verifies and parses ----------------------------------------------


@pytest.fixture(scope="session")
def cache() -> Path:
    return Path("~/.cache/autan-vv").expanduser()


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("name", [c.name for c in list_cases()])
def test_every_file_downloads_verifies_and_parses(name: str, cache: Path) -> None:
    case = get_case(name)
    for f, path in zip(case.files, fetch(case, cache), strict=True):
        parsed = parse(f, path)
        assert parsed is not None
        if f.parser == "plot3d2d":
            ((x, _),) = parsed
            dims = [
                int(v)
                for v in path.name.replace("-", "x").split("_")[-1].split(".")[0].split("x")
                if v.isdigit()
            ]
            if len(dims) == 2:
                assert x.shape == (dims[1], dims[0])


def _zone(tables: list[Table], title: str) -> Table:
    return next(t for t in tables if t.title == title)


@pytest.mark.network
def test_tmr_page_values_match_the_data_files(cache: Path) -> None:
    fp = get_case("tmr-2dzp-flatplate")
    fetch(fp, cache, roles=("reference",))
    ref = fp.reference_values
    cf = read_tecplot(cache / fp.name / "cf_convergence.dat")
    cd = read_tecplot(cache / fp.name / "drag_convergence.dat")
    for code in ("CFL3D", "FUN3D"):
        zone_cf, zone_cd = _zone(cf, code), _zone(cd, code)
        finest = int(np.argmax(zone_cf.column("N")))
        assert zone_cf.column("N")[finest] == 208896
        assert zone_cf.column("C_f,x=0.97")[finest] == pytest.approx(
            ref["cf_x0.97"][code], rel=1e-6
        )
        assert zone_cd.column("C_D")[finest] == pytest.approx(ref["cd"][code], rel=1e-6)

    n12 = get_case("tmr-2dn00-naca0012")
    fetch(n12, cache, roles=("reference",))
    (clcd,) = read_tecplot(cache / n12.name / "n0012clcd_cfl3d_sa.dat")
    for alpha, cl in n12.reference_values["cl"].items():
        row = int(np.argmin(np.abs(clcd.column("alpha, deg") - float(alpha))))
        assert clcd.column("cl")[row] == pytest.approx(cl, rel=1e-9, abs=1e-12)
        assert clcd.column("cd")[row] == pytest.approx(n12.reference_values["cd"][alpha], rel=1e-9)


@pytest.mark.network
def test_rae2822_case_9_conditions_decode_from_the_1981_tape(cache: Path) -> None:
    rae = get_case("rae2822")
    path = fetch(rae, cache, roles=("experiment",))[-1]
    table = read_tape_1981(path)[1122]
    row = table[np.argmin(np.abs(table[:, 0] - 9.0))]
    # Cook, McDonald and Firmin (AGARD AR-138), case 9: M = 0.73, alpha = 3.19 deg (2.79 corrected),
    # Re = 6.5e6; the tape stores about four significant digits.
    assert row[1] == pytest.approx(0.73, abs=1e-4)
    assert row[2] == pytest.approx(3.19, abs=1e-3)
    assert row[3] == pytest.approx(rae.conditions["case_9"]["alpha_deg"], abs=1e-3)
    assert row[4] == pytest.approx(6.5e6, rel=1e-4)
    case9 = rae.reference_values["case_9"]
    assert row[7:10] == pytest.approx([case9["cn"], case9["cm"], case9["cd"]], abs=1e-4)
