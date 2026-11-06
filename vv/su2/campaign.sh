#!/usr/bin/env bash
# Run SU2 on prepared cases one after the other, inside the autan-su2 image (the spark-5359
# campaign lane runs this script in the image with the prepared root mounted at /work):
#   vv/su2/campaign.sh THREADS IMAGE_ID CASE...      (CASE relative to /work, e.g. m080-a125/n1024)
# Same outputs and stop rules as run.sh: log.txt and provenance.json in each case directory.
set -uo pipefail

threads=$1
image_id=$2
# SU2's threads are its OpenMP threads (-t); no library may start one per core behind --cpus.
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
shift 2
cd /work || exit 1

stop_when_done() {
  # SU2 measures a residual's drop from its largest value, the protocol from its first: SIGTERM
  # SU2 (it ends the iteration and writes its output) once every rms[...] of history.csv is ORDERS
  # below its first value, or once the worst equation's drop has not grown by 0.1 orders in STALL
  # iterations (with STALL_SKIP_ZERO=1, the worst of those that did not start identically zero).
  local pid=$1 history=$2
  while kill -0 "$pid" 2>/dev/null; do
    if [ -f "$history" ] && awk -F, -v orders="${ORDERS:-10}" -v stall="${STALL:-2000}" -v skipzero="${STALL_SKIP_ZERO:-0}" \
          -v stopfile="${history%/*}/stop.csv" '
        BEGIN {
          window = 500; rtol = 1e-7
          # The stop.csv of a case (#493) holds log10 thresholds per rms column, the outputs to keep
          # stationary, the window and the tolerance; without it, the relative criterion applies.
          while ((getline line < stopfile) > 0) {
            split(line, f, ",")
            if (f[1] == "threshold") limit[f[2]] = f[3] + 0
            if (f[1] == "output") want[f[2]] = 1
            if (f[1] == "window") window = f[2] + 0
            if (f[1] == "rtol") rtol = f[2] + 0
            absolute = 1
          }
        }
        NR == 1 {
          nf = NF
          for (i = 1; i <= NF; i++) {
            name = $i; gsub(/[" ]/, "", name)
            if (name ~ /^rms\[/) { col[i] = 1; if (name in limit) level[i] = limit[name]; else if (absolute) bad = 1 }
            if (name in want) { out[i] = 1; seen[name] = 1 }
          }
          for (name in want) if (!(name in seen)) bad = 1
          next
        }
        # SU2 writes the history in buffered blocks: its last line may be cut short.
        NF < nf { next }
        absolute {
          # Every residual at or below its threshold, the outputs stationary; the stall rule counts only
          # the equations still above their threshold.
          above = 0; staller = -1e300
          for (i in col) if ($i + 0 > level[i]) { above = 1; if ($i - level[i] > staller) staller = $i - level[i] }
          for (i in out) val[NR, i] = $i + 0
          if (!above) started = 0
          else if (!started || staller < ref - 0.1) { ref = staller; since = NR; started = 1 }
          stalled = above && NR - since >= stall
          last = NR; lastabove = above
          next
        }
        NR == 2 {
          top = -1e300
          for (i in col) { first[i] = $i + 0; if (first[i] > top) top = first[i] }
          # case-schema.md: an equation starting 12 orders below the largest is identically zero,
          # exempt while it stays at or below that floor.
          for (i in col) if (first[i] < top - 12) zero[i] = 1
        }
        {
          worst = -1e300; staller = -1e300
          for (i in col) {
            if (zero[i] && $i + 0 > top - 12) risen[i] = 1
            if (zero[i] && !risen[i]) continue
            if ($i + 0 - first[i] > worst) worst = $i + 0 - first[i]
            if (!(skipzero && zero[i]) && $i + 0 - first[i] > staller) staller = $i + 0 - first[i]
          }
          # 1e-6 order stricter, so that the linear-space test of autan_vv.su2.history agrees.
          if (worst <= -orders - 1e-6) found = 1
          best[NR] = (NR == 2 || staller < best[NR - 1]) ? staller : best[NR - 1]
          if (NR > stall + 2 && best[NR - stall] - best[NR] < 0.1) found = 1
        }
        END {
          # A stall is judged at the last row, so a run that recovers is not stopped for an old one.
          if (absolute && stalled) found = 1
          if (absolute && !bad && last && !lastabove && last - 1 > window) {
            still = 1
            for (i in out) {
              scale = val[last, i] < 0 ? -val[last, i] : val[last, i]
              for (j = last - window; j < last; j++) {
                d = val[j, i] - val[last, i]
                if ((d < 0 ? -d : d) > rtol * scale) still = 0
              }
            }
            if (still) found = 1
          }
          exit !found
        }' "$history"; then
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
  stop_sha=""
  if [ -f "$dir/stop.csv" ]; then stop_sha=$(sha256sum "$dir/stop.csv" | cut -d' ' -f1); fi
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
  "stop": {"orders": ${ORDERS:-10}, "stall": ${STALL:-2000}, "stall_skip_zero": ${STALL_SKIP_ZERO:-0}, "stop_csv_sha256": "$stop_sha"},
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
