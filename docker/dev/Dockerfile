# syntax=docker/dockerfile:1.7
# Development and CI image: CUDA 13.0, GCC 13, CMake, Ninja, CUDA-aware Open MPI over UCX,
# METIS, PT-Scotch, parallel HDF5, CGNS, OSU micro-benchmarks, Python 3.12 with uv.
# Libraries built from source live in /opt/autan. See docs/infra/image.md.

ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config git ca-certificates curl \
      libmetis-dev zlib1g-dev libibverbs-dev librdmacm-dev libnuma-dev \
      python3 python3.12 python3.12-dev python3.12-venv \
      openssh-client openssh-server jq \
 && rm -rf /var/lib/apt/lists/*

FROM base AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends bzip2 flex bison autoconf automake libtool \
 && rm -rf /var/lib/apt/lists/*
ARG PREFIX=/opt/autan
ENV PATH=${PREFIX}/bin:${PATH}
WORKDIR /src
COPY fetch.sh /usr/local/bin/fetch

# libcuda.so is injected by the NVIDIA container runtime, not present at build time: link against the stub.
ENV CUDA_STUBS=/usr/local/cuda/lib64/stubs

ARG UCX_VERSION=1.22.0
ARG UCX_SHA256=258941cddd14ca60d38c0d31b9b09ec1052c901086841011a498da8b55a3cb24
RUN fetch https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz ${UCX_SHA256} \
 && cd ucx-${UCX_VERSION} \
 && LDFLAGS="-L${CUDA_STUBS}" ./contrib/configure-release --prefix=${PREFIX} \
      --with-cuda=/usr/local/cuda --with-verbs --with-rdmacm --without-java --without-go \
      --disable-doxygen-doc --enable-mt \
 && make -j"$(nproc)" && make install && cd .. && rm -rf ucx-*

ARG OMPI_VERSION=5.0.11
ARG OMPI_SHA256=e668a3c4acd50c41dc204c8a6dd98a611e0f26af89cf677577fa9be8a2698003
# CUDA components are built as plugins so the same libmpi runs on hosts without a GPU driver
# (Open MPI packaging guide, "CUDA-aware support").
RUN fetch https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VERSION}.tar.bz2 ${OMPI_SHA256} \
 && cd openmpi-${OMPI_VERSION} \
 && ./configure --prefix=${PREFIX} \
      --with-cuda=/usr/local/cuda --with-cuda-libdir=${CUDA_STUBS} \
      --with-ucx=${PREFIX} \
      --enable-mca-dso=accelerator-cuda,btl-smcuda,rcache-gpusm,rcache-rgpusm,coll-cuda \
      --disable-sphinx --disable-oshmem --without-ofi \
 && make -j"$(nproc)" && make install && cd .. && rm -rf openmpi-*

ARG HDF5_VERSION=2.1.1
ARG HDF5_SHA256=efff93b5a904d66e8f626d7da60b5eedc9faf544be27dbabbaa87967b8ad798b
RUN fetch https://github.com/HDFGroup/hdf5/releases/download/${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz ${HDF5_SHA256} \
 && cmake -S hdf5-${HDF5_VERSION} -B hdf5-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_C_COMPILER=mpicc -DHDF5_ENABLE_PARALLEL=ON \
      -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_JAVA=OFF \
      -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=ON -DBUILD_TESTING=OFF \
      -DHDF5_ENABLE_ZLIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
 && cmake --build hdf5-build && cmake --install hdf5-build && rm -rf hdf5-*

ARG CGNS_VERSION=4.5.2
ARG CGNS_SHA256=95075e1fd0b51d97b1b96b73ebe03b1a551fbcc9cd2b2b6f487ccccedcff5964
RUN fetch https://github.com/CGNS/CGNS/archive/refs/tags/v${CGNS_VERSION}.tar.gz ${CGNS_SHA256} \
 && cmake -S CGNS-${CGNS_VERSION} -B cgns-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_C_COMPILER=mpicc -DCMAKE_PREFIX_PATH=${PREFIX} \
      -DCGNS_ENABLE_HDF5=ON -DHDF5_NEED_MPI=ON -DCGNS_ENABLE_PARALLEL=ON \
      -DCGNS_ENABLE_64BIT=ON -DCGNS_BUILD_SHARED=ON -DCGNS_ENABLE_FORTRAN=OFF \
      -DCGNS_ENABLE_TESTS=OFF -DCGNS_BUILD_CGNSTOOLS=OFF \
 && cmake --build cgns-build && cmake --install cgns-build && rm -rf CGNS-* cgns-*

# SCOTCH_Num is 64-bit so graphs with more than 2^31 edges partition; the METIS compatibility
# library is not built, so METIS symbols come only from the system METIS.
ARG SCOTCH_VERSION=7.0.17
ARG SCOTCH_SHA256=d8bc97540e83520f5218ebc6ed44ee370e0faa1e9c68d5e3c4cd8db0c683d48b
RUN fetch https://gitlab.inria.fr/scotch/scotch/-/archive/v${SCOTCH_VERSION}/scotch-v${SCOTCH_VERSION}.tar.gz ${SCOTCH_SHA256} \
 && cmake -S scotch-v${SCOTCH_VERSION} -B scotch-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_C_COMPILER=mpicc -DBUILD_SHARED_LIBS=ON \
      -DBUILD_PTSCOTCH=ON -DINTSIZE=64 -DBUILD_LIBSCOTCHMETIS=OFF -DINSTALL_METIS_HEADERS=OFF \
      -DBUILD_LIBESMUMPS=OFF -DBUILD_FORTRAN=OFF -DENABLE_TESTS=OFF \
 && cmake --build scotch-build && cmake --install scotch-build && rm -rf scotch-*

ARG OSU_VERSION=7.5.2
ARG OSU_SHA256=618de3d0b1122f73a9229177d2da1e5cd62e431190580cb915f2605849cbbbdc
RUN fetch https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz ${OSU_SHA256} \
 && cd osu-micro-benchmarks-${OSU_VERSION} \
 && ./configure --prefix=${PREFIX}/osu CC=mpicc CXX=mpicxx \
      --enable-cuda --with-cuda=/usr/local/cuda LDFLAGS="-L${CUDA_STUBS}" \
 && make -j"$(nproc)" && make install && cd .. && rm -rf osu-*

FROM base AS dev
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=build /opt/autan /opt/autan
ARG AUTAN_IMAGE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/Remenby31/autan" \
      org.opencontainers.image.description="Autan development and CI image" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${AUTAN_IMAGE_REVISION}"
ENV PATH=/opt/autan/bin:/opt/autan/osu/libexec/osu-micro-benchmarks/mpi/pt2pt:/opt/autan/osu/libexec/osu-micro-benchmarks/mpi/collective:${PATH} \
    CMAKE_PREFIX_PATH=/opt/autan \
    CMAKE_GENERATOR=Ninja \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/tmp/uv-cache \
    AUTAN_IMAGE_REVISION=${AUTAN_IMAGE_REVISION}
RUN echo /opt/autan/lib > /etc/ld.so.conf.d/autan.conf && ldconfig
