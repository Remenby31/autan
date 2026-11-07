# NEVER push this image to a registry or publish it: it holds a GPL-3.0 binary, and distributing it
# would oblige us to provide its corresponding source. Build it locally and keep it local.
#
# Construct2D, GPL-3.0, offline tool, not redistributed: the RAE 2822 grid generator of
# docs/validation/meshes.md, built and run in its own image, never vendored in this repository nor
# linked with Autan. Source: C2D_URL below, pinned by C2D_SHA256 and checked before the build.
# The release's Makefile_Linux_MacOSX unchanged, in a fixed directory: the binary is then
# byte-reproducible (its debug information records the path) and writes the pinned grids.
ARG BASE_IMAGE=autan-dev:local
FROM ${BASE_IMAGE}
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gfortran-13=13.3.0-6ubuntu2~24.04.1 libgfortran-13-dev=13.3.0-6ubuntu2~24.04.1 \
      make unzip curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG C2D_URL=https://downloads.sourceforge.net/project/construct2d/Construct2D_2.1.4.zip
ARG C2D_SHA256=9926dd6c5abb4c675826261a6c39ac00537f6bcbeccc25024f1e1bc6d0bf3159
WORKDIR /opt/construct2d
RUN curl -fsSL -o release.zip "${C2D_URL}" \
 && echo "${C2D_SHA256}  release.zip" | sha256sum -c - \
 && unzip -q release.zip \
 && make -C Construct2D_2.1.4 -f Makefile_Linux_MacOSX FC=gfortran-13 \
 && install -m 0755 Construct2D_2.1.4/construct2d /usr/local/bin/construct2d
WORKDIR /work
