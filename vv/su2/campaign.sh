#!/usr/bin/env bash
# Run SU2 on prepared cases one after the other, inside the autan-su2 image (the spark-5359
# campaign lane runs this script in the image with the prepared root mounted at /work):
#   vv/su2/campaign.sh THREADS IMAGE_ID CASE...      (CASE relative to /work, e.g. m080-a125/n1024)
# Same outputs and stop rules as run.sh: log.txt and provenance.json in each case directory.
set -uo pipefail

threads=$1
image_id=$2
shift 2
cd /work

stop_when_done() {
  # SU2 measures a residual's drop from its largest value, the protocol from its first: SIGTERM
  # SU2 (it ends the iteration and writes its output) once every rms[...] of history.csv is ORDERS
  # below its first value, or once the worst equation's drop has not grown by 0.1 orders in STALL
  # iterations.
  local pid=$1 history=$2
  while kill -0 "$pid" 2>/dev/null; do
    if [ -f "$history" ] && awk -F, -v orders="${ORDERS:-10}" -v stall="${STALL:-2000}" '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /"rms\[/) col[i] = 1; next }
        NR == 2 {
          top = -1e300
          for (i in col) { first[i] = $i + 0; if (first[i] > top) top = first[i] }
          # case-schema.md: an equation starting 12 orders below the largest is identically zero,
          # exempt while it stays at or below that floor.
          for (i in col) if (first[i] < top - 12) zero[i] = 1
        }
        {
          worst = -1e300
          for (i in col) {
            if (zero[i] && $i + 0 > top - 12) risen[i] = 1
            if (zero[i] && !risen[i]) continue
            if ($i + 0 - first[i] > worst) worst = $i + 0 - first[i]
          }
          # 1e-6 order stricter, so that the linear-space test of autan_vv.su2.history agrees.
          if (worst <= -orders - 1e-6) found = 1
          best[NR] = (NR == 2 || worst < best[NR - 1]) ? worst : best[NR - 1]
          if (NR > stall + 2 && best[NR - stall] - best[NR] < 0.1) found = 1
        }
        END { exit !found }' "$history"; then
      kill -TERM "$pid"
      return
    fi
    sleep 5
  done
}

for case in "$@"; do
  dir=/work/$case
  mesh=$(sed -n 's/^MESH_FILENAME= //p' "$dir/case.cfg")
  case_sha=$(sha256sum "$dir/case.cfg" | cut -d' ' -f1)
  mesh_sha=$(sha256sum "$dir/$mesh" | cut -d' ' -f1)
  rm -f "$dir/history.csv" "$dir/provenance.json"
  start=$(date +%s.%N)
  (cd "$dir" && exec nice -n 10 SU2_CFD -t "$threads" case.cfg >log.txt 2>&1) &
  pid=$!
  stop_when_done "$pid" "$dir/history.csv"
  wait "$pid"
  status=$?
  end=$(date +%s.%N)
  cat >"$dir/provenance.json" <<JSON
{
  "schema": "autan.su2-run/1",
  "image": "autan-su2:v8.6.0-omp",
  "image_id": "$image_id",
  "su2": "$(cat /opt/su2/VERSION)",
  "threads": $threads,
  "case_sha256": "$case_sha",
  "mesh_sha256": "$mesh_sha",
  "cpu": "$(lscpu | sed -n 's/^Model name: *//p' | sort -u | paste -sd '+')",
  "wall_seconds": $(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }'),
  "exit_status": $status
}
JSON
  echo "$case exit $status"
done
echo campaign-done
