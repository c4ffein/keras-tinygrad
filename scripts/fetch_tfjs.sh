#!/usr/bin/env bash
# Fetch the tf.js build the browser hub embeds — pinned version, sha256
# verified, so the 1.4 MB file stays out of git (experiments/*/tf.min.js are
# gitignored) yet every rebuild embeds exactly the bytes that were measured.
#   scripts/fetch_tfjs.sh            # -> experiments/m0-keras-trainstep/tf.min.js
set -euo pipefail
VERSION=4.22.0
SHA256=300dfae273d20b4046f46a06d735688f03675a807561e9bcb5f664eb2f3d2831
DEST=$(cd "$(dirname "$0")/.." && pwd)/experiments/m0-keras-trainstep/tf.min.js
URL="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@$VERSION/dist/tf.min.js"
if [ -f "$DEST" ] && echo "$SHA256  $DEST" | sha256sum -c --quiet 2>/dev/null; then
  echo "tf.min.js $VERSION already present and verified"; exit 0
fi
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
curl -fsSL "$URL" -o "$tmp"
echo "$SHA256  $tmp" | sha256sum -c --quiet || { echo "sha256 mismatch for $URL — refusing to use it"; exit 1; }
mv "$tmp" "$DEST"; echo "fetched tf.js $VERSION -> $DEST (sha256 verified)"
