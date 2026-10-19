#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_STUDY="${SOURCE_STUDY:-$REPO_ROOT/artifacts/token_equal_seed103/20261017_192348}"
STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/sequential_forget_retain/$(date +%Y%m%d_%H%M%S)}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
MODE="${MODE:-run}"
COMMON_KL="${COMMON_KL:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$REPO_ROOT"
CONTROL="$SCRIPT_DIR/sequential_forget_retain.py"
ARGS=(--study "$STUDY_DIR" --source-study "$SOURCE_STUDY"
  --forget-epochs "${FORGET_EPOCHS:-10}" --retain-epochs "${RETAIN_EPOCHS:-10}"
  --retain-epoch-size "${RETAIN_EPOCH_SIZE:-200}" --learning-rate "${LEARNING_RATE:-1e-5}"
  --forget-common-kl-weight "${FORGET_COMMON_KL_WEIGHT:-0}")
if [[ -n "${PAIRED_STUDY:-}" ]]; then ARGS+=(--paired-study "$PAIRED_STUDY"); fi
if [[ "$COMMON_KL" == 0 ]]; then ARGS+=(--no-common-kl)
elif [[ "$COMMON_KL" != 1 ]]; then printf 'COMMON_KL must be 0 or 1\n' >&2; exit 2; fi
if [[ "$MODE" == dry-run ]]; then
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode dry-run "${ARGS[@]}"
  exit
elif [[ "$MODE" != run ]]; then printf 'MODE must be run or dry-run\n' >&2; exit 2; fi
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode prepare "${ARGS[@]}"
mkdir -p "$STUDY_DIR/logs"
trap 'rc=$?; printf "Stopped (exit %s); measurements preserved: %s\n" "$rc" "$STUDY_DIR" >&2; exit "$rc"' ERR
"$PYTHON_BIN" "$SCRIPT_DIR/seed103_branch_experiment.py" --mode check-gpus \
  --gpus "$FOUR_GPUS" --min-free-gib "${MIN_FREE_GIB:-40}" > "$STUDY_DIR/logs/gpu_before.json"
CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 "$CONTROL" --mode train --study "$STUDY_DIR" \
  2>&1 | tee "$STUDY_DIR/logs/train.log"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode analyze --study "$STUDY_DIR" \
  2>&1 | tee "$STUDY_DIR/logs/analysis.log"
printf 'Completed: %s\n' "$STUDY_DIR"
