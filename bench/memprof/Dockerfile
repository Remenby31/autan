# autan-memprof's image: the dev image with gdb, for the stacks (see docs/infra/bench.md).
#   docker build -t autan-memprof:local bench/memprof
FROM autan-dev:local
USER root
RUN apt-get update -q && apt-get install -y -q --no-install-recommends gdb \
    && rm -rf /var/lib/apt/lists/*
