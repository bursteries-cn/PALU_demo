#!/usr/bin/env bash
# Bounded six-update seed103 replays. Does not launch the three-branch study.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
: "${STUDY_DIR:?Set STUDY_DIR to the existing frozen seed103 study}"
: "${PROFILE_DIR:?Set PROFILE_DIR to a new output directory}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p -- "$PROFILE_DIR"
cd -- "$REPO_ROOT"
for backend in baseline trace; do
    if [[ -e "$PROFILE_DIR/$backend" ]]; then
        echo "Refusing to replace profiling output: $PROFILE_DIR/$backend" >&2
        exit 2
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/seed103_branch_experiment.py" --mode check-gpus \
        --gpus "$FOUR_GPUS" --min-free-gib 40 > "$PROFILE_DIR/${backend}_gpu_before.json"
    CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=4 "$SCRIPT_DIR/profile_seed103_training.py" \
        --study "$STUDY_DIR" --output "$PROFILE_DIR/$backend" --backend "$backend" \
        > "$PROFILE_DIR/$backend.log" 2>&1
done
printf 'Both bounded profiling runs completed: %s\n' "$PROFILE_DIR"
