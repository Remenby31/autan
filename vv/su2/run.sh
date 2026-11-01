#!/usr/bin/env bash
# Run SU2 on one prepared case in the autan-su2 image, at low priority.
#   vv/su2/run.sh ROOT CASE THREADS
# ROOT is the directory `autan-vv su2 prepare` wrote (mounted whole: the meshes are shared), CASE
# a case directory under it (e.g. m080-a125/n0256). Writes CASE/log.txt and CASE/provenance.json.
set -euo pipefail

root=$(cd "$1" && pwd)
case=$2
threads=$3
image=${AUTAN_SU2_IMAGE:-autan-su2:v8.6.0-omp}
dir="$root/$case"

image_id=$(docker image inspect "$image" --format '{{.Id}}')
version=$(docker run --rm "$image" cat /opt/su2/VERSION)
start=$(date +%s.%N)
status=0
mesh=$(sed -n 's/^MESH_FILENAME= //p' "$dir/case.cfg")
case_sha=$(sha256sum "$dir/case.cfg" | cut -d' ' -f1)
mesh_sha=$(sha256sum "$dir/$mesh" | cut -d' ' -f1)
rm -f "$dir/history.csv" "$dir/provenance.json"
name="autan-su2-$$"
docker run --rm --name "$name" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e OMP_NUM_THREADS="$threads" --cpus "$threads" -v "$root:/work" -w "/work/$case" "$image" \
  nice -n 10 SU2_CFD -t "$threads" case.cfg >"$dir/log.txt" 2>&1 &
run=$!
# SU2 measures a residual's drop from its largest value, the protocol from its first: stop SU2
# (SIGTERM: it ends the iteration and writes its output) once every rms[...] of history.csv is
# ORDERS below its first value, so that it does not run on at its round-off floor, or once the
# residuals have stalled: the worst equation's drop has not grown by 0.1 orders in STALL iterations.
while kill -0 "$run" 2>/dev/null; do
  if [ -f "$dir/history.csv" ] && awk -F, -v orders="${ORDERS:-10}" -v stall="${STALL:-2000}" '
      NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /"rms\[/) col[i] = 1; next }
      NR == 2 { for (i in col) first[i] = $i + 0 }
      {
        worst = -1e300
        for (i in col) if ($i + 0 - first[i] > worst) worst = $i + 0 - first[i]
        if (worst <= -orders) found = 1
        best[NR] = (NR == 2 || worst < best[NR - 1]) ? worst : best[NR - 1]
        if (NR > stall + 2 && best[NR - stall] - best[NR] < 0.1) found = 1
      }
      END { exit !found }' "$dir/history.csv"; then
    docker kill --signal TERM "$name" >/dev/null 2>&1 || true
    break
  fi
  sleep 5
done
wait "$run" || status=$?
end=$(date +%s.%N)

cat >"$dir/provenance.json" <<JSON
{
  "schema": "autan.su2-run/1",
  "image": "$image",
  "image_id": "$image_id",
  "su2": "$version",
  "threads": $threads,
  "case_sha256": "$case_sha",
  "mesh_sha256": "$mesh_sha",
  "cpu": "$(lscpu | sed -n 's/^Model name: *//p' | sort -u | paste -sd '+')",
  "wall_seconds": $(echo "$end - $start" | bc -l | sed 's/^\./0./'),
  "exit_status": $status
}
JSON
exit "$status"
