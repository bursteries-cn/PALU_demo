#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
# Fixed comparison arm: inherited sample-run settings must not change its policy.
# Resource/location controls (MODE, FOUR_GPUS, PYTHON_BIN, SOURCE_STUDY, STUDY_DIR)
# remain configurable through the shared launcher.
export SEEDS="105"
export LEARNING_RATE="5e-6"
export MAX_UPDATES="63"
export STOPPING_SCOPE="global"
export PROBE_INTERVAL="epoch"
export THRESHOLD_COMPARISON="lt"
export THRESHOLD_LOW="0.4"
export THRESHOLD_HIGH="0.5"
export ZERO_WEIGHT_RETAIN_NO_GRAD="1"
export STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/global_forget_seed105/$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/run_forget_until_threshold.sh"
