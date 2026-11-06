# GitHub Actions runner for the DGX Spark: the dev image plus the runner agent, running as an
# unprivileged user. Jobs execute directly in this container (no Docker socket). See
# docs/infra/runners.md for the threat model.
ARG DEV_IMAGE
FROM ${DEV_IMAGE}

ARG RUNNER_VERSION=2.338.2
ARG RUNNER_SHA256_ARM64=670adb2f268fe21dcbb58b3db0e0aa9d5b3bddfe2c651481336e561ff86de945
RUN test "$(dpkg --print-architecture)" = arm64 \
 && useradd --create-home --uid 1001 --shell /bin/bash runner \
 && mkdir -p /opt/actions-runner && cd /opt/actions-runner \
 && curl -fsSL --retry 5 -o runner.tgz \
      https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz \
 && echo "${RUNNER_SHA256_ARM64}  runner.tgz" | sha256sum -c - \
 && tar -xzf runner.tgz && rm runner.tgz \
 && apt-get update && ./bin/installdependencies.sh \
 && apt-get install -y -q --no-install-recommends gdb && rm -rf /var/lib/apt/lists/* \
 && chown -R runner:runner /opt/actions-runner

# gdb for the hang watchdog (bench/ci/ctest-watchdog.sh), and the library it preloads into the
# tests it runs so that gdb may attach to them under the hosts' Yama ptrace_scope 1 (ptracer.c).
COPY ptracer.c /tmp/ptracer.c
RUN install -d /opt/autan-runner/lib \
 && cc -shared -fPIC -O2 -o /opt/autan-runner/lib/libautan_ptracer.so /tmp/ptracer.c \
 && chmod 0555 /opt/autan-runner/lib/libautan_ptracer.so && rm /tmp/ptracer.c
ENV AUTAN_WATCHDOG_PTRACER=/opt/autan-runner/lib/libautan_ptracer.so

COPY --chmod=0555 job-started.sh entrypoint.sh /opt/autan-runner/
ARG REPOSITORY=Remenby31/autan
# The host memory a job waits for before its first step, in GiB (0: none); root-owned, like the
# repository, so no workflow can change it.
ARG MIN_AVAILABLE_GIB=0
RUN echo "${REPOSITORY}" > /opt/autan-runner/repository \
 && echo "${MIN_AVAILABLE_GIB}" > /opt/autan-runner/min-available-gib \
 && install -d -o runner -g runner /state /work
ENV ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/autan-runner/job-started.sh
USER runner
WORKDIR /opt/actions-runner
ENTRYPOINT ["/opt/autan-runner/entrypoint.sh"]
CMD ["run"]
