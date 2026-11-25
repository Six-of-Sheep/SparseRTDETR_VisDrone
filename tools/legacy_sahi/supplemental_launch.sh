#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
: "${SUPPLEMENTAL_PYTHON:?explicit certified interpreter required}"
entry="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/supplemental_job.py"
exec "$SUPPLEMENTAL_PYTHON" -B "$entry" start --contract "$1"
