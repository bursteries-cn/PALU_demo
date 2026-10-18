#!/usr/bin/env bash
# P336 / P421 / P386, three new seeds each. No training in --dry-run/--prepare-only.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/overshoot_seed_replicates/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-201,202,203}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$PYTHON_BIN" ]]; then echo "Missing PALU Python: $PYTHON_BIN" >&2; exit 2; fi
MODE=run
case "${1:-}" in
    --dry-run) MODE=dry-run ;;
    --prepare-only) MODE=prepare ;;
    --analyze-only) MODE=analyze ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run|--prepare-only|--analyze-only]" >&2; exit 2 ;;
esac
ARGS=(--mode "$MODE" --study "$STUDY_DIR" --gpus "$GPUS" --seeds "$SEEDS" --min-free-gib "${MIN_FREE_GIB:-78}")
if [[ -n "${MODEL_PATH:-}" ]]; then ARGS+=(--model-path "$MODEL_PATH"); fi
if [[ -n "${RETAIN_LOGS:-}" ]]; then ARGS+=(--retain-logs "$RETAIN_LOGS"); fi
if [[ -n "${TOFU_DATA_PATH:-}" ]]; then ARGS+=(--tofu-data "$TOFU_DATA_PATH"); fi
cd -- "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$SCRIPT_DIR/overshoot_seed_replicates.py" "${ARGS[@]}"
