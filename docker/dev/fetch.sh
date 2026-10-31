#!/bin/sh
# Usage: fetch URL SHA256. Downloads URL, verifies its SHA-256 and extracts it into the current directory.
set -eu
url=$1
sum=$2
file=$(basename "$url")
curl -fsSL --retry 5 -o "$file" "$url"
echo "$sum  $file" | sha256sum -c -
tar -xf "$file"
rm -f "$file"
