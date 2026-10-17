#!/usr/bin/env bash
# One uninterrupted Full-to-update60 trajectory; no fork or repair branch.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
: "${SOURCE_STUDY:?Set SOURCE_STUDY to the validated frozen seed103 study}"
: "${STUDY_DIR:?Set STUDY_DIR to a new output directory}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$REPO_ROOT"
CONTROL="$SCRIPT_DIR/seed103_token_equal.py"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode prepare \
  --study "$STUDY_DIR" --source-study "$SOURCE_STUDY"
mkdir -p "$STUDY_DIR/logs"
trap 'rc=$?; printf "Stopped (exit %s); artifacts preserved: %s\n" "$rc" "$STUDY_DIR" >&2; exit "$rc"' ERR
"$PYTHON_BIN" "$SCRIPT_DIR/seed103_branch_experiment.py" --mode check-gpus \
  --gpus "$FOUR_GPUS" --min-free-gib 40 > "$STUDY_DIR/logs/gpu_before.json"
CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 "$CONTROL" --mode train --study "$STUDY_DIR" \
  2>&1 | tee "$STUDY_DIR/logs/train.log"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode analyze --study "$STUDY_DIR"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$SCRIPT_DIR/plot_seed_epoch.py" "$STUDY_DIR/run" \
  --output "$STUDY_DIR/plots" > "$STUDY_DIR/logs/plots.log" 2>&1
printf 'Completed: %s\n' "$STUDY_DIR"
