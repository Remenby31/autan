#!/usr/bin/env bash
# TMR 2DZP flat plate, SA-neg: one case per mesh of physics/tmr/plot3d.py, run through autan-run.
# Usage: physics/tmr/flatplate.sh MESHDIR OUT [PRECISION] [MAX_ITERATIONS]
# Conditions (tmbwg.github.io/turbmodels/flatplate.html): M 0.2, Re 5e6 per unit length, T 300 K,
# plate from x = 0 to 2 at y = 0, adiabatic; total pressure and temperature at the inflow, the
# freestream static pressure at the outflow, farfield on top, symmetry ahead of the plate.
set -euo pipefail
meshes=$1 out=$2 precision=${3:-fp64} iterations=${4:-20000}
turbulence=${TURBULENCE:-"{ model = \"sa\", nu_tilde_ratio = 3.0 }"}
plate=${PLATE:-wall}
span=${SPAN:-1.0}  # z extent of the meshes of plot3d.py (--span)
top=$(git rev-parse --show-toplevel)
build=${AUTAN_BUILD:-$top/build}
export AUTAN_RUN=$build/runtime/autan-run
# p and T of the freestream from Re, M and Sutherland's law (case-schema.md closure), then totals.
read -r p0 t0 pinf <<<"$(python3 -c "
g, R, T, M, Re = 1.4, 8.314462618 / 0.0289647, 300.0, 0.2, 5.0e6
mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
V = M * (g * R * T) ** 0.5
p = Re * mu / V * R * T
f = 1 + 0.5 * (g - 1) * M * M
print(repr(p * f ** (g / (g - 1))), repr(T * f), repr(p))")"
mkdir -p "$out"
for msh in "$meshes"/*.msh; do
  name=$(basename "$msh" .msh)
  dir="$out/$name"
  mkdir -p "$dir"
  "$build/io/autan-mesh-info" --json "$msh" >"$dir/mesh-info.json"
  cat >"$dir/case.toml" <<TOML
schema = "autan.case/1"
name = "tmr-2dzp-$name"
mesh = "$msh"

[physics]
kind = "compressible"
precision = "$precision"
turbulence = $turbulence
gas = { kind = "ideal-gas", name = "air", molar_mass = "0.0289647 kg / mol", gamma = 1.4, prandtl = 0.72, viscosity = { law = "sutherland", mu_ref = "1.716e-05 Pa * s", t_ref = "273.15 K", s = "110.4 K" } }

[freestream]
mach = 0.2
aoa = "0.0 deg"
re = 5000000.0
ref_length = "1.0 m"
temperature = "300.0 K"

[time]
kind = "steady"
max_iterations = $iterations
residual_orders = 10.0

[[boundaries]]
zones = "plate"
condition = { kind = "$plate" }

[[boundaries]]
zones = "symmetry"
condition = { kind = "symmetry" }

[[boundaries]]
zones = "sides"
condition = { kind = "symmetry" }

[[boundaries]]
zones = "inflow"
condition = { kind = "inlet", total_pressure = "$p0 Pa", total_temperature = "$t0 K" }

[[boundaries]]
zones = "outflow"
condition = { kind = "outlet", static_pressure = "$pinf Pa" }

[[boundaries]]
zones = "top"
condition = { kind = "farfield" }

[[monitors]]
kind = "forces"
name = "plate"
surfaces = "plate"
every = 1
ref_area = "$(python3 -c "print(2.0 * $span)") m ** 2"
TOML
  uv run --locked --project "$top" autan run "$dir/case.toml" --mesh-info "$dir/mesh-info.json" --backend cpu --runs "$out/runs" --json |
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['path'], d['run']['status'], d['run']['iterations'])"
done
