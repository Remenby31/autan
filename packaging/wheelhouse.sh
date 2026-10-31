#!/bin/sh
# Builds an offline wheelhouse for one platform: install.sh, the pinned uv archive, the Python
# build that uv picks for AUTAN_PYTHON (laid out as a uv Python mirror) and every wheel autan
# needs. On the air-gapped machine: AUTAN_OFFLINE=1 AUTAN_FIND_LINKS=<dir> sh <dir>/install.sh
#
# Usage: sh packaging/wheelhouse.sh OUT TARGET WHEELS
#   TARGET  x86_64-unknown-linux-gnu | aarch64-unknown-linux-gnu | aarch64-apple-darwin
#   WHEELS  directory holding the autan wheel to bundle
set -eu
out=$1
target=$2
wheels=$3
root=$(cd "$(dirname "$0")/.." && pwd)
python=${AUTAN_PYTHON:-3.13}

case "$target" in
  x86_64-unknown-linux-gnu) sha_var=UV_SHA256_X86_64_LINUX key=linux-x86_64-gnu ;;
  aarch64-unknown-linux-gnu) sha_var=UV_SHA256_AARCH64_LINUX key=linux-aarch64-gnu ;;
  aarch64-apple-darwin) sha_var=UV_SHA256_AARCH64_DARWIN key=macos-aarch64-none ;;
  *) echo "wheelhouse.sh: unknown target $target" >&2 && exit 2 ;;
esac
# The pins live in install.sh only; read them from there.
uv_version=$(sed -n 's/^UV_VERSION="\(.*\)"$/\1/p' "$root/install.sh")
sha=$(sed -n "s/^$sha_var=\"\(.*\)\"$/\1/p" "$root/install.sh")

mkdir -p "$out/python"
cp "$root/install.sh" "$out/"
archive=$out/uv-$target.tar.gz
curl --proto '=https' --tlsv1.2 -fsSL -o "$archive" \
  "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$target.tar.gz"
echo "$sha  $archive" | sha256sum -c -

# Ask the same uv the installer will run which Python build it picks, so the mirror matches.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
host=$(uname -m)-$(uname -s)
case "$host" in
  x86_64-Linux) host_target=x86_64-unknown-linux-gnu ;;
  aarch64-Linux) host_target=aarch64-unknown-linux-gnu ;;
  arm64-Darwin) host_target=aarch64-apple-darwin ;;
  *) echo "wheelhouse.sh: cannot run uv on $host" >&2 && exit 2 ;;
esac
curl --proto '=https' --tlsv1.2 -fsSL "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$host_target.tar.gz" |
  tar -xz -C "$tmp"
url=$("$tmp/uv-$host_target/uv" python list "$python" --only-downloads --all-platforms --all-arches --show-urls --output-format json |
  python3 -c "import json, sys
rows = [r for r in json.load(sys.stdin)
        if r['key'].startswith('cpython-') and r['key'].endswith('-$key') and r['url'] and r['variant'] == 'default']
print(rows[0]['url'])")
# Mirror layout: <release tag>/<file name>, the part of the URL after /releases/download/.
path=$(printf '%s\n' "${url#*/releases/download/}" | sed 's/%2B/+/g')
mkdir -p "$out/python/$(dirname "$path")"
curl --proto '=https' --tlsv1.2 -fsSL -o "$out/python/$path" "$url"

# pip matches --platform tags literally: list every tag a glibc 2.28 or macOS 14 machine accepts
# that dependencies actually publish.
case "$target" in
  x86_64-*linux*) set -- manylinux_2_28_x86_64 manylinux_2_27_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64 ;;
  aarch64-*linux*) set -- manylinux_2_28_aarch64 manylinux_2_27_aarch64 manylinux_2_17_aarch64 manylinux2014_aarch64 ;;
  *) set -- macosx_14_0_arm64 macosx_13_0_arm64 macosx_12_0_arm64 macosx_11_0_arm64 ;;
esac
platforms=""
for tag in "$@"; do platforms="$platforms --platform $tag"; done
abi=cp$(echo "$python" | tr -d .)
# shellcheck disable=SC2086 # $platforms is a list of options
uvx pip download --quiet --only-binary :all: --dest "$out" --find-links "$wheels" $platforms \
  --python-version "$python" --implementation cp --abi "$abi" --abi abi3 --abi none autan
ls "$out"
