#!/usr/bin/env bash
# PALU / Llama3.1 / seed103: old single, new single, original four-GPU reduction.
# Run from any directory. No training is performed by --dry-run or --prepare-only.
# Example: SINGLE_GPU=2 FOUR_GPUS=2,3,4,5 bash scripts/palu/run_layout_loss_curves_seed103.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/layout_loss_curves_seed103/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
SINGLE_GPU="${SINGLE_GPU:-0}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
MIN_FREE_GIB="${MIN_FREE_GIB:-78}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN. Set PYTHON_BIN to the PALU environment's Python." >&2
    exit 2
fi
if [[ "$SINGLE_GPU" == *,* ]]; then
    echo "SINGLE_GPU must identify exactly one GPU." >&2
    exit 2
fi
IFS=',' read -r -a FOUR_GPU_ARRAY <<< "$FOUR_GPUS"
if [[ "${#FOUR_GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "FOUR_GPUS must list exactly four distinct GPU IDs." >&2
    exit 2
fi
for ((i=0; i<4; i++)); do
    if [[ -z "${FOUR_GPU_ARRAY[i]}" ]]; then echo "Empty GPU ID in FOUR_GPUS." >&2; exit 2; fi
    for ((j=i+1; j<4; j++)); do
        if [[ "${FOUR_GPU_ARRAY[i]}" == "${FOUR_GPU_ARRAY[j]}" ]]; then
            echo "FOUR_GPUS must use four distinct GPU IDs." >&2
            exit 2
        fi
    done
done
STUDY_DIR="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$STUDY_DIR")"
CONTROL="$REPO_ROOT/scripts/palu/layout_loss_curve_experiment.py"
TRAINING="$REPO_ROOT/scripts/palu/layout_token_training.py"
PREPARE=("$PYTHON_BIN" "$CONTROL" --mode prepare --study "$STUDY_DIR")
if [[ -n "${MODEL_PATH:-}" ]]; then PREPARE+=(--model-path "$MODEL_PATH"); fi
if [[ -n "${RETAIN_LOGS:-}" ]]; then PREPARE+=(--retain-logs "$RETAIN_LOGS"); fi

case "${1:-}" in
    --dry-run)
        PREPARE[3]=dry-run
        "${PREPARE[@]}"
        printf '\nSINGLE_GPU=%s FOUR_GPUS=%s\n' "$SINGLE_GPU" "$FOUR_GPUS"
        exit 0 ;;
    --analyze-only)
        "$PYTHON_BIN" "$CONTROL" --mode analyze --study "$STUDY_DIR"
        exit 0 ;;
    --prepare-only|"") ;;
    *) echo "Usage: $0 [--dry-run|--prepare-only|--analyze-only]" >&2; exit 2 ;;
esac

cd -- "$REPO_ROOT"
"${PREPARE[@]}"
if [[ "${1:-}" == --prepare-only ]]; then exit 0; fi
mkdir -p -- "$STUDY_DIR/logs"
printf 'Experiment: %s\n' "$STUDY_DIR"
printf 'Each actual epoch ends with immediate in-memory evaluation, validation and intermediate-weight deletion before the next training microbatch. Final models and all observations are retained.\n'
trap 'rc=$?; printf "Stopped (exit %s). Measurements preserved in %s\n" "$rc" "$STUDY_DIR" >&2; exit "$rc"' ERR

for ARM in old_single new_single four_native; do
    OUTPUT="$STUDY_DIR/$ARM"
    STATUS="$("$PYTHON_BIN" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/"manifest.json"; print(json.loads(p.read_text())["status"] if p.is_file() else "new")' "$OUTPUT")"
    if [[ "$STATUS" == completed ]]; then
        printf 'Skip completed arm: %s\n' "$ARM"
        continue
    fi
    if [[ "$STATUS" != new ]]; then
        echo "Arm $ARM is incomplete (status=$STATUS). Use a new STUDY_DIR; live optimizer/pending-gradient resume is not implemented." >&2
        exit 3
    fi
    if [[ "$STATUS" == new ]]; then
        LAYOUT=single
        NORMALIZATION=native
        PADDING=microbatch
        GPUS="$SINGLE_GPU"
        if [[ "$ARM" == new_single ]]; then NORMALIZATION=global; PADDING=update; fi
        if [[ "$ARM" == four_native ]]; then LAYOUT=four; GPUS="$FOUR_GPUS"; fi
        "$PYTHON_BIN" "$CONTROL" --mode check-gpus --gpus "$GPUS" --min-free-gib "$MIN_FREE_GIB" | tee "$STUDY_DIR/logs/${ARM}_gpu_before_train.log"
        TRAIN_ARGS=("$TRAINING" --mode train --config "$STUDY_DIR/frozen/config.json"
                    --schedule "$STUDY_DIR/frozen/schedule.pt" --output "$OUTPUT" --seed 103
                    --layout "$LAYOUT" --normalization "$NORMALIZATION" --padding-scope "$PADDING"
                    --online-evaluation --delete-evaluated-intermediate)
        if [[ "$LAYOUT" == four ]]; then
            CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 "${TRAIN_ARGS[@]}" 2>&1 | tee "$STUDY_DIR/logs/${ARM}_train.log"
        else
            CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" "${TRAIN_ARGS[@]}" 2>&1 | tee "$STUDY_DIR/logs/${ARM}_train.log"
        fi
    fi
done
"$PYTHON_BIN" "$CONTROL" --mode analyze --study "$STUDY_DIR" 2>&1 | tee "$STUDY_DIR/logs/analysis.log"
printf 'Completed. Curves and CSV files: %s/analysis\n' "$STUDY_DIR"
