#!/bin/sh
# Installs Autan for the current user: curl -LsSf https://remenby31.github.io/autan/install.sh | sh
#
# Needs only curl or wget, tar and a sha256 tool. Fetches a pinned uv, checks its SHA-256, and
# runs `uv tool install autan`; uv downloads Python when the system lacks the version asked for. Nothing is
# written outside the user's home and nothing asks for sudo.
#
# Environment:
#   AUTAN_VERSION          version to install (default: latest)
#   AUTAN_PYTHON           Python version for Autan's environment (default: 3.13; >= 3.12)
#   AUTAN_GPU              auto | cuda | cpu (default: auto, only reported: the Linux wheel
#                          carries the CUDA code and the GPU is chosen at run time)
#   AUTAN_FIND_LINKS       directory of wheels searched before PyPI
#   AUTAN_OFFLINE          set to 1 to install from the wheelhouse in AUTAN_FIND_LINKS only:
#                          uv-<target>.tar.gz, python/ (a uv Python mirror) and every wheel
#   AUTAN_NO_MODIFY_PATH   set to 1 to leave shell startup files untouched
#
# The whole script is one function called on the last line, so a truncated download runs
# nothing.

set -u

UV_VERSION="0.13.6"
UV_SHA256_X86_64_LINUX="f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"
UV_SHA256_AARCH64_LINUX="229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d"
UV_SHA256_AARCH64_DARWIN="60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472"
# manylinux_2_28 wheels: glibc 2.28 or newer.
GLIBC_MIN_MINOR=28

say() { printf 'autan-install: %s\n' "$*" >&2; }
die() {
  say "error: $*"
  exit 1
}

platform() {
  os=$(uname -s)
  arch=$(uname -m)
  case "$os" in
    Linux)
      case "$arch" in
        x86_64 | amd64) triple=x86_64-unknown-linux-gnu sha=$UV_SHA256_X86_64_LINUX ;;
        aarch64 | arm64) triple=aarch64-unknown-linux-gnu sha=$UV_SHA256_AARCH64_LINUX ;;
        *) die "no Autan build for Linux $arch; supported: x86_64, aarch64" ;;
      esac
      if ldd --version 2>&1 | grep -qi musl; then
        die "musl libc (Alpine) is not supported; use a glibc distribution or the container image ghcr.io/remenby31/autan-runtime"
      fi
      glibc=$(ldd --version 2>/dev/null | head -n1 | sed -n 's/.* \([0-9][0-9]*\)\.\([0-9][0-9]*\)$/\2/p')
      if [ -n "$glibc" ] && [ "$glibc" -lt "$GLIBC_MIN_MINOR" ]; then
        die "glibc 2.$glibc is too old; Autan needs glibc 2.$GLIBC_MIN_MINOR (RHEL 8, Ubuntu 20.04, Debian 11 or newer)"
      fi
      ;;
    Darwin)
      # uname -m reports x86_64 under Rosetta; ask the kernel for the real CPU.
      if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
        triple=aarch64-apple-darwin sha=$UV_SHA256_AARCH64_DARWIN
      else
        die "Intel Macs are not supported; Autan needs Apple Silicon"
      fi
      major=$(sw_vers -productVersion | cut -d. -f1)
      [ "$major" -ge 14 ] || die "macOS $major is too old; Autan needs macOS 14 or newer"
      ;;
    MINGW* | MSYS* | CYGWIN*) die "Windows is not supported natively; run this script inside WSL2" ;;
    *) die "unsupported operating system: $os" ;;
  esac
}

gpu_report() {
  wanted=${AUTAN_GPU:-auto}
  case "$wanted" in auto | cuda | cpu) ;; *) die "AUTAN_GPU=$wanted: expected auto, cuda or cpu" ;; esac
  if [ "$os" != Linux ]; then
    [ "$wanted" != cuda ] || die "AUTAN_GPU=cuda: CUDA needs Linux with an NVIDIA GPU"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    say "NVIDIA GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
  elif [ "$wanted" = cuda ]; then
    die "AUTAN_GPU=cuda but nvidia-smi finds no GPU; install the NVIDIA driver (580 or newer) first"
  elif [ -e /dev/kfd ]; then
    say "AMD GPU found: ROCm builds are not published yet, Autan will run on the CPU"
  else
    say "no GPU driver found: Autan will run on the CPU"
  fi
}

fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then
    wget --https-only --secure-protocol=TLSv1_2 -q "$1" -O "$2"
  else
    die "neither curl nor wget is installed"
  fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    die "no sha256sum or shasum to verify the download; install coreutils"
  fi
}

# A private uv, so the user's own uv (or its absence) is never changed.
install_uv() {
  data=${XDG_DATA_HOME:-$HOME/.local/share}/autan
  uv=$data/uv-$UV_VERSION/uv
  [ -x "$uv" ] && return 0
  tmp=$(mktemp -d) || die "mktemp failed"
  archive=$tmp/uv.tar.gz
  if [ "${AUTAN_OFFLINE:-0}" = 1 ]; then
    cp "$AUTAN_FIND_LINKS/uv-$triple.tar.gz" "$archive" || die "the wheelhouse has no uv-$triple.tar.gz"
  else
    say "downloading uv $UV_VERSION"
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$triple.tar.gz" "$archive" ||
      die "could not download uv"
  fi
  actual=$(sha256_of "$archive")
  [ "$actual" = "$sha" ] || die "uv checksum mismatch: expected $sha, got $actual"
  tar -xzf "$archive" -C "$tmp" || die "could not unpack uv"
  mkdir -p "$data/uv-$UV_VERSION" || die "could not create $data"
  mv "$tmp/uv-$triple/uv" "$uv" || die "could not install uv to $data"
  rm -rf "$tmp"
}

install_autan() {
  spec=autan
  [ -n "${AUTAN_VERSION:-}" ] && spec="autan==$AUTAN_VERSION"
  set -- tool install --force --python "${AUTAN_PYTHON:-3.13}" "$spec"
  [ -n "${AUTAN_FIND_LINKS:-}" ] && set -- "$@" --find-links "$AUTAN_FIND_LINKS"
  if [ "${AUTAN_OFFLINE:-0}" = 1 ]; then
    set -- "$@" --offline --no-index
    # --offline also forbids Python downloads, even from a local mirror: install it first.
    UV_PYTHON_INSTALL_MIRROR="file://$AUTAN_FIND_LINKS/python" \
      "$uv" python install "${AUTAN_PYTHON:-3.13}" || die "the wheelhouse has no Python ${AUTAN_PYTHON:-3.13}"
  fi
  say "installing $spec"
  "$uv" "$@" || die "uv could not install $spec"
}

update_path() {
  bin=$("$uv" tool dir --bin) || die "uv tool dir failed"
  case ":$PATH:" in *":$bin:"*) return 0 ;; esac
  if [ "${AUTAN_NO_MODIFY_PATH:-0}" = 1 ]; then
    say "add $bin to your PATH to run autan"
  elif [ -n "${GITHUB_PATH:-}" ]; then
    echo "$bin" >>"$GITHUB_PATH"
  else
    if "$uv" tool update-shell >/dev/null 2>&1; then
      say "restart your shell, or run: export PATH=\"$bin:\$PATH\""
    else
      say "add $bin to your PATH to run autan"
    fi
  fi
}

main() {
  [ "$(id -u)" -ne 0 ] || say "warning: running as root installs Autan for root only"
  if [ "${AUTAN_OFFLINE:-0}" = 1 ]; then
    [ -d "${AUTAN_FIND_LINKS:-}" ] || die "AUTAN_OFFLINE=1 needs AUTAN_FIND_LINKS, the wheelhouse directory"
    AUTAN_FIND_LINKS=$(cd "$AUTAN_FIND_LINKS" && pwd)
  fi
  platform
  gpu_report
  install_uv
  install_autan
  update_path
  say "checking the installation (autan doctor)"
  "$bin/autan" doctor || say "autan doctor reported problems; each line above says how to fix it"
}

main "$@" || exit 1
