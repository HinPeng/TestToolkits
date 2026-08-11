#!/usr/bin/env bash
# Create a text-focused evidence archive.  Generated binaries and caches are
# deliberately excluded; cache_tree.txt remains available for cache analysis.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 EVIDENCE_DIR ARCHIVE_PATH" >&2
  exit 2
fi

EVIDENCE_DIR="$(cd "$1" && pwd)"
ARCHIVE_PATH="$2"
mkdir -p "$(dirname "$ARCHIVE_PATH")"
ARCHIVE_BASENAME="$(basename "$ARCHIVE_PATH")"

tar \
  --exclude="./$ARCHIVE_BASENAME" \
  --exclude='./cache' \
  --exclude='*.so' \
  --exclude='*.o' \
  --exclude='*.a' \
  --exclude='*.cubin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.bin' \
  -czf "$ARCHIVE_PATH" \
  -C "$EVIDENCE_DIR" .

echo "wrote $ARCHIVE_PATH"
