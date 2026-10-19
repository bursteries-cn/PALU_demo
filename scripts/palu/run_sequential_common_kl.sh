#!/usr/bin/env bash
# Matched seed103 control: F + 2 common-KL -> 2 R + 2 common-KL.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export FORGET_COMMON_KL_WEIGHT=2 COMMON_KL=1
export PAIRED_STUDY="${PAIRED_STUDY:-$REPO_ROOT/artifacts/sequential_forget_retain/20261019_205517}"
export STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/sequential_common_kl/$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/run_sequential_forget_retain.sh"
