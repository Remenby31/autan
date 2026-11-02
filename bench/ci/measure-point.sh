#!/usr/bin/env bash
# One bisection probe, run in the gpu-bench job on the Spark that detected the regression:
# builds the loop benchmark of SHA in a separate worktree, checks the machine with STREAM against
# its baseline, then measures the tracked loops of SHA and gates them, both with bench/ci/gate.sh
# (one confirmation rerun), exactly as the detection did. Results go to OUT.
# Usage: bench/ci/measure-point.sh SHA OUT   (from the repository root; meshes/hex-n128.msh and
#        build/bench/stream present)
# Exit: 0 the loops pass, 1 confirmed regression, 4 STREAM failed three times (the machine is not
# fit to measure: the point is not interpreted), 2 any other failure.
set -uo pipefail
sha=$1 out=$2
mkdir -p "$out"
tree=${RUNNER_TEMP:-/tmp}/autan-bisect-$sha
bench=(uv run --project bench --locked autan-bench)

git worktree add --force --detach "$tree" "$sha" >/dev/null || exit 2
trap 'git worktree remove --force "$tree" >/dev/null 2>&1' EXIT
# Under the GPU lock: on the bench Spark no measurement runs beside a compilation.
bench/lock/autan-gpu-lock -m "gpu-bench: build a bisection probe" -t 600 -- bash -c "
  cmake -S '$tree' -B '$tree/build' -DAUTAN_ENABLE_CUDA=ON -DAUTAN_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121 && cmake --build '$tree/build' --target autan_loop_bench
" >"$out/build.log" 2>&1 || exit 2

stream=(stream --binary build/bench/stream --out "$out/stream.json")
healthy=0
for _ in 1 2 3; do
  "${bench[@]}" "${stream[@]}" || exit 2
  bench/ci/gate.sh "$out/stream.json" bench/baselines/gb10.json -- "${bench[@]}" "${stream[@]}"
  case $? in
    0) healthy=1; break ;;
    1) ;;
    *) exit 2 ;;
  esac
done
((healthy)) || exit 4

loops=(run --suite codegen --binary "$tree/build/codegen/autan_loop_bench" --precision fp32
  --out "$out/codegen-hex-n128-fp32.json" -- meshes/hex-n128.msh --policy fp32 --order rcm)
"${bench[@]}" "${loops[@]}" || exit 2
bench/ci/gate.sh "$out/codegen-hex-n128-fp32.json" bench/baselines/gb10-codegen-hex-n128-fp32.json \
  -- "${bench[@]}" "${loops[@]}"
case $? in
  0) exit 0 ;;
  1) exit 1 ;;
  *) exit 2 ;;
esac
