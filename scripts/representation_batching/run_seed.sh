#!/usr/bin/env bash
# Usage: bash scripts/representation_batching/run_seed.sh 1 [--dry-run] [--config PATH]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ $# -eq 0 ]]; then
    read -r -p "请输入 seed: " SEED_VALUE
    set -- "${SEED_VALUE}"
fi
exec "${PYTHON_BIN:-python3}" scripts/representation_batching/run_seed.py "$@"
