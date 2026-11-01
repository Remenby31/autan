# syntax=docker/dockerfile:1.7
# SU2 (LGPL-2.1), an external tool for the comparison of docs/validation/su2.md: built and run
# in its own image, never vendored in this repository nor linked with Autan.
ARG BASE_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential python3 git ca-certificates pkg-config libopenmpi-dev openmpi-bin \
 && rm -rf /var/lib/apt/lists/*

ARG SU2_TAG=v8.6.0
ARG SU2_COMMIT=7048fd35f56d4bf52e586429d6746a36a10255f8
ARG JOBS=8
WORKDIR /src
RUN git clone --depth 1 --branch ${SU2_TAG} https://github.com/su2code/SU2.git \
 && cd SU2 \
 && test "$(git rev-parse HEAD)" = "${SU2_COMMIT}" \
 && ./meson.py setup build --prefix=/opt/su2 --buildtype=release \
      -Dwith-mpi=enabled -Dwith-omp=false -Denable-autodiff=false -Denable-directdiff=false \
      -Denable-pywrapper=false -Denable-tecio=false -Denable-cgns=false \
 && nice -n 10 ./ninja -C build -j ${JOBS} install \
 && cp build/meson-logs/meson-log.txt /opt/su2/meson-log.txt \
 && cp LICENSE.md /opt/su2/LICENSE.md \
 && printf '%s %s\n' "${SU2_TAG}" "${SU2_COMMIT}" > /opt/su2/VERSION \
 && cd / && rm -rf /src/SU2

ENV PATH=/opt/su2/bin:${PATH}
WORKDIR /work
