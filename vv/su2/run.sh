#!/usr/bin/env bash
# Run SU2 on one prepared case in the autan-su2 image, at low priority.
#   vv/su2/run.sh ROOT CASE RANKS
# ROOT is the directory `autan-vv su2 prepare` wrote (mounted whole: the meshes are shared), CASE
# a case directory under it (e.g. m080-a125/n0256). Writes CASE/log.txt and CASE/provenance.json.
set -euo pipefail

root=$(cd "$1" && pwd)
case=$2
ranks=$3
image=${AUTAN_SU2_IMAGE:-autan-su2:v8.6.0}
dir="$root/$case"

image_id=$(docker image inspect "$image" --format '{{.Id}}')
version=$(docker run --rm "$image" cat /opt/su2/VERSION)
start=$(date +%s.%N)
status=0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --cpus "$ranks" \
  -v "$root:/work" -w "/work/$case" "$image" \
  nice -n 10 mpirun -np "$ranks" --bind-to none SU2_CFD case.cfg >"$dir/log.txt" 2>&1 || status=$?
end=$(date +%s.%N)

cat >"$dir/provenance.json" <<JSON
{
  "schema": "autan.su2-run/1",
  "image": "$image",
  "image_id": "$image_id",
  "su2": "$version",
  "ranks": $ranks,
  "cpu": "$(lscpu | sed -n 's/^Model name: *//p' | sort -u | paste -sd '+')",
  "wall_seconds": $(echo "$end - $start" | bc -l | sed 's/^\./0./'),
  "exit_status": $status
}
JSON
exit "$status"
