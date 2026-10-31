"""`autan-vv` command line: manufactured solutions, mesh families and order studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autan_vv.errors import VVError


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def _levels(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(","))


def _cmd_mms(args: argparse.Namespace) -> None:
    from autan_vv.mms import CATALOG, dump_json, to_cpp  # noqa: PLC0415

    names = sorted(CATALOG) if args.name == "all" else [args.name]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        ms = CATALOG[name]()
        (args.out / f"{name}.hpp").write_text(to_cpp(ms))
        (args.out / f"{name}.json").write_text(dump_json(ms))
        _say(f"{name}: {args.out / name}.hpp, {args.out / name}.json")


def _cmd_meshes(args: argparse.Namespace) -> None:
    from autan_vv.meshes.families import generate_family  # noqa: PLC0415

    family, _ = generate_family(
        args.kind, args.levels, args.out, perturbation=args.perturbation, seed=args.seed
    )
    for lv in family.levels:
        _say(f"{lv.file}: {lv.n_cells} cells, h = {lv.h:.4g}, sha256 {lv.file_sha256[:12]}")


def _cmd_order(args: argparse.Namespace) -> None:
    from autan_vv.study import order_study, write_report  # noqa: PLC0415

    data = json.loads(args.errors.read_text())
    report = order_study(data.get("label", args.errors.stem), int(data["dim"]), data["levels"])
    write_report(report, args.out, args.plot)
    for norm, fit in report["fits"].items():
        _say(f"{norm}: p = {fit['order']:.3f}, finest pair {fit['pairwise_orders'][-1]:.3f}")


def _cmd_verify(args: argparse.Namespace) -> None:
    from autan_vv.verify import advection_diffusion_study  # noqa: PLC0415

    report = advection_diffusion_study(
        args.kind,
        args.levels,
        args.out,
        perturbation=args.perturbation,
        seed=args.seed,
        precision=args.precision,
    )
    for norm, fit in report["fits"].items():
        _say(f"{norm}: p = {fit['order']:.3f}, finest pair {fit['pairwise_orders'][-1]:.3f}")


def main(argv: list[str] | None = None) -> int:
    """Run the `autan-vv` CLI; returns the process exit code (2 on a coded V&V error)."""
    parser = argparse.ArgumentParser(prog="autan-vv", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("mms", help="export manufactured solutions to C++ and JSON")
    p.add_argument("name", help="solution name or 'all'")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_mms)

    p = sub.add_parser("meshes", help="generate a seeded mesh family of the unit cube")
    p.add_argument("kind", choices=("tet", "hex", "prism", "hybrid", "poly", "agglomerated"))
    p.add_argument("--levels", type=_levels, default=(8, 16, 32, 64))
    p.add_argument("--perturbation", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_meshes)

    p = sub.add_parser(
        "order", help="observed order from errors on a mesh family (JSON in, JSON out)"
    )
    p.add_argument("errors", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plot", type=Path)
    p.set_defaults(func=_cmd_order)

    p = sub.add_parser(
        "verify", help="reference advection-diffusion solve and order study on a family"
    )
    p.add_argument("kind", choices=("tet", "hex", "prism", "hybrid", "poly", "agglomerated"))
    p.add_argument("--levels", type=_levels, default=(8, 16, 24, 32))
    p.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    p.add_argument("--perturbation", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except VVError as err:
        sys.stderr.write(f"{err}\n")
        return 2
    return 0
