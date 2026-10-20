#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# These defaults define the requested new experiment; old launchers stay unchanged.
export SEEDS="${SEEDS:-103 104 105}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export MAX_UPDATES="${MAX_UPDATES:-63}"
export STOPPING_SCOPE="${STOPPING_SCOPE:-sample_permanent}"
export PROBE_INTERVAL="${PROBE_INTERVAL:-epoch}"
export THRESHOLD_COMPARISON="${THRESHOLD_COMPARISON:-lt}"
export ZERO_WEIGHT_RETAIN_NO_GRAD="${ZERO_WEIGHT_RETAIN_NO_GRAD:-1}"
export STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/sample_forget_threshold/$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/run_forget_until_threshold.sh"
