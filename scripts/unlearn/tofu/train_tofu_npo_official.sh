#!/usr/bin/env bash
set -euo pipefail

# Only the launcher lives in PALU_demo. Training, configs and evaluation come
# from --upstream-dir, whose clean HEAD is checked against the pinned commit.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${NPO_LAUNCHER_PYTHON:-python3}" "${SCRIPT_DIR}/run_official_npo.py" "$@"
