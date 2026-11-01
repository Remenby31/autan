#!/bin/bash
# ONERA M6 n=12 stall: diagnostics variants (experiment build, AUTAN_EXPERIMENT_DIAG).
set -u
cd /src
B=/src/build-m6
export UV_PROJECT_ENVIRONMENT=/src/.venv-m6 UV_CACHE_DIR=/tmp/uv-cache
cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DAUTAN_EXPERIMENT_DIAG=ON >/dev/null && cmake --build $B -j 8 --target autan-run >/dev/null || exit 1
uv sync --locked -q
D=$B/m6; R=$D/exp4; mkdir -p $R
uv run --no-sync autan prepare $D/cases/onera-m6-n012.toml --mesh-info $D/n012.meshinfo.json > $D/base.json || exit 1
python3 - $D/base.json $R <<'PY'
import json,sys
b=json.load(open(sys.argv[1])); out=sys.argv[2]
def w(name, edit):
    d=json.loads(json.dumps(b)); edit(d); json.dump(d, open(f"{out}/{name}.json","w"))
def steady(n):
    def e(d): d["time"]["max_iterations"]=n
    return e
w("base", steady(1500))
w("mono300", steady(1500))
PY
run() { name=$1; shift; rm -rf $R/$name; env "$@" AUTAN_EXP_DUMP=$R/$name nice $B/runtime/autan-run --input $R/$name.json --runs $R/$name > $R/$name.log 2>&1; }
run base X=1 &
run mono300 AUTAN_EXP_MONO=300 &
wait
echo done
