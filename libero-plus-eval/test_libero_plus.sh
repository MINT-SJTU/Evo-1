#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${1:-libero_spatial}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$SCRIPT_DIR"

for category in camera robot language light layout background noise; do
  python -m "evo_libero_plus_clients.${category}" --suite "$SUITE" "$@"
done