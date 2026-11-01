#!/usr/bin/env bash
# TMR SA-neg cases (tmbwg.github.io/turbmodels), one run per mesh of physics/tmr/plot3d.py, through
# autan-run. Usage: physics/tmr/case.sh CASE MESHDIR OUT [PRECISION] [MAX_ITERATIONS] [ALPHA_DEG]
#   flatplate  2DZP: M 0.2, Re 5e6 per metre; total pressure and temperature at the inflow,
#              freestream static pressure at the outflow, farfield on top, symmetry ahead of the
#              adiabatic plate (x 0 to 2); forces over the plate.
#   bump       2DBC: M 0.2, Re 3e6 per metre; same inflow and outflow, symmetry on top and on
#              y = 0 outside the adiabatic wall (x 0 to 1.5); forces over the wall.
#   naca0012   2DN00: M 0.15, Re 6e6 per chord, farfield 500 chords away, adiabatic airfoil.
# T = 300 K throughout; the meshes are one layer of thickness $SPAN (default 1) between `sides`.
set -euo pipefail
kind=$1 meshes=$2 out=$3 precision=${4:-fp64} iterations=${5:-20000} alpha=${6:-0}
turbulence=${TURBULENCE:-"{ model = \"sa\", nu_tilde_ratio = 3.0 }"}
span=${SPAN:-1.0}
top=$(git rev-parse --show-toplevel)
build=${AUTAN_BUILD:-$top/build}
export AUTAN_RUN=$build/runtime/autan-run
case $kind in
  flatplate) mach=0.2 re=5.0e6 wall=plate length=2.0 ;;
  bump) mach=0.2 re=3.0e6 wall=wall length=1.0 ;;
  naca0012) mach=0.15 re=6.0e6 wall=airfoil length=1.0 ;;
  *) echo "unknown case $kind" >&2; exit 2 ;;
esac
# p and T of the freestream from Re, M and Sutherland's law (case-schema.md closure), then totals.
read -r p0 t0 pinf <<<"$(python3 -c "
g, R, T, M, Re = 1.4, 8.314462618 / 0.0289647, 300.0, $mach, $re
mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
V = M * (g * R * T) ** 0.5
p = Re * mu / V * R * T
f = 1 + 0.5 * (g - 1) * M * M
print(repr(p * f ** (g / (g - 1))), repr(T * f), repr(p))")"
rule() { printf '[[boundaries]]\nzones = "%s"\ncondition = %s\n\n' "$1" "$2"; }
boundaries() {
  rule sides '{ kind = "symmetry" }'
  rule "$wall" '{ kind = "wall" }'
  case $kind in
    naca0012) rule farfield '{ kind = "farfield" }' ;;
    *)
      rule symmetry '{ kind = "symmetry" }'
      rule inflow "{ kind = \"inlet\", total_pressure = \"$p0 Pa\", total_temperature = \"$t0 K\" }"
      rule outflow "{ kind = \"outlet\", static_pressure = \"$pinf Pa\" }"
      if [ "$kind" = flatplate ]; then rule top '{ kind = "farfield" }'; else rule top '{ kind = "symmetry" }'; fi
      ;;
  esac
}
mkdir -p "$out"
for msh in "$meshes"/*.msh; do
  name=$(basename "$msh" .msh)
  dir="$out/$name"
  mkdir -p "$dir"
  "$build/io/autan-mesh-info" --json "$msh" >"$dir/mesh-info.json"
  cat >"$dir/case.toml" <<TOML
schema = "autan.case/1"
name = "tmr-$kind-$name"
mesh = "$msh"

[physics]
kind = "compressible"
precision = "$precision"
turbulence = $turbulence
gas = { kind = "ideal-gas", name = "air", molar_mass = "0.0289647 kg / mol", gamma = 1.4, prandtl = 0.72, viscosity = { law = "sutherland", mu_ref = "1.716e-05 Pa * s", t_ref = "273.15 K", s = "110.4 K" } }

[freestream]
mach = $mach
aoa = "$alpha deg"
re = $re
ref_length = "1.0 m"
temperature = "300.0 K"

[time]
kind = "steady"
max_iterations = $iterations
residual_orders = 10.0

$(boundaries)

[[monitors]]
kind = "forces"
name = "wall"
surfaces = "$wall"
every = 1
ref_area = "$(python3 -c "print($length * $span)") m ** 2"
TOML
  uv run --locked --project "$top" autan run "$dir/case.toml" --mesh-info "$dir/mesh-info.json" --backend cpu --runs "$out/runs" --json |
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d['path'], d['run']['status'], d['run']['iterations'])"
done
