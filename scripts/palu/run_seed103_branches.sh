#!/usr/bin/env bash
# Four GPUs, whole-update token means, common step-six optimizer fork.
# Run from any directory. The four DeepSpeed jobs execute sequentially.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/seed103_branches/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
# The original 78 GiB single-GPU guard does not apply to four-rank ZeRO3.
# This is a free-memory guard, not a guarantee that all evaluation kernels fit.
MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN. Set PYTHON_BIN to the PALU environment." >&2
    exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "$FOUR_GPUS"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "FOUR_GPUS must list exactly four distinct GPU IDs." >&2
    exit 2
fi
for ((i=0; i<4; i++)); do
    if [[ -z "${GPU_ARRAY[i]}" || "${GPU_ARRAY[i]}" == *[[:space:]]* ]]; then
        echo "FOUR_GPUS contains an empty or whitespace-bearing GPU ID." >&2
        exit 2
    fi
    for ((j=i+1; j<4; j++)); do
        if [[ "${GPU_ARRAY[i]}" == "${GPU_ARRAY[j]}" ]]; then
            echo "FOUR_GPUS must use four distinct GPU IDs." >&2
            exit 2
        fi
    done
done
STUDY_DIR="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$STUDY_DIR")"
CONTROL="$REPO_ROOT/scripts/palu/seed103_branch_experiment.py"
TRAINING="$REPO_ROOT/scripts/palu/seed103_branch_training.py"
PREPARE=("$PYTHON_BIN" "$CONTROL" --mode prepare --study "$STUDY_DIR")
if [[ -n "${MODEL_PATH:-}" ]]; then PREPARE+=(--model-path "$MODEL_PATH"); fi
if [[ -n "${RETAIN_LOGS:-}" ]]; then PREPARE+=(--retain-logs "$RETAIN_LOGS"); fi
case "${1:-}" in
    --dry-run)
        PREPARE[3]=dry-run
        "${PREPARE[@]}"
        printf '\nFOUR_GPUS=%s MIN_FREE_GIB=%s\n' "$FOUR_GPUS" "$MIN_FREE_GIB"
        exit 0 ;;
    --analyze-only)
        "$PYTHON_BIN" "$CONTROL" --mode analyze --study "$STUDY_DIR" --cleanup-fork
        exit 0 ;;
    --prepare-only|"") ;;
    *) echo "Usage: $0 [--dry-run|--prepare-only|--analyze-only]" >&2; exit 2 ;;
esac
cd -- "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="" "${PREPARE[@]}"
if [[ "${1:-}" == --prepare-only ]]; then exit 0; fi
mkdir -p -- "$STUDY_DIR/logs"
printf 'Experiment: %s\n' "$STUDY_DIR"
printf 'Prefix: 6 complete updates / 192 pairs. Branches: gamma=1,0,0.1; each ends at update60.\n'
printf 'Actual epoch evaluation is immediate. Only verified intermediate weights are deleted; final models and all observations are retained.\n'
trap 'rc=$?; printf "Stopped (exit %s). Results/restart files preserved in %s\n" "$rc" "$STUDY_DIR" >&2; exit "$rc"' ERR

status_of() {
    "$PYTHON_BIN" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/"manifest.json"; print(json.loads(p.read_text())["status"] if p.is_file() else "new")' "$1"
}
run_phase() {
    local phase="$1" arm="${2:-}" stage="$1" label="$1"
    local args=("$TRAINING" --phase "$phase" --study "$STUDY_DIR")
    if [[ "$phase" == branch ]]; then
        stage="$arm"
        label="$arm"
        args+=(--arm "$arm")
    fi
    "$PYTHON_BIN" "$CONTROL" --mode check-storage --study "$STUDY_DIR" --stage "$stage" | tee "$STUDY_DIR/logs/${label}_disk_before.log"
    "$PYTHON_BIN" "$CONTROL" --mode check-gpus --gpus "$FOUR_GPUS" --min-free-gib "$MIN_FREE_GIB" | tee "$STUDY_DIR/logs/${label}_gpu_before.log"
    CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=4 "${args[@]}" 2>&1 | tee "$STUDY_DIR/logs/${label}_train.log"
}
PREFIX_STATUS="$(status_of "$STUDY_DIR/prefix")"
if [[ "$PREFIX_STATUS" == new ]]; then
    run_phase prefix
elif [[ "$PREFIX_STATUS" == prefix_completed ]]; then
    printf 'Reuse completed common prefix. Fork tensors are verified again before branch loading.\n'
else
    echo "Prefix incomplete (status=$PREFIX_STATUS). Preserve this folder and use a fresh STUDY_DIR; arbitrary partial-step resume is not implemented." >&2
    exit 3
fi
for ARM in continue repair weak; do
    STATUS="$(status_of "$STUDY_DIR/$ARM")"
    if [[ "$STATUS" == completed ]]; then
        printf 'Skip completed branch: %s\n' "$ARM"
    elif [[ "$STATUS" == new ]]; then
        run_phase branch "$ARM"
    else
        echo "Branch $ARM is incomplete (status=$STATUS). Preserve it and use a fresh STUDY_DIR; arbitrary partial-step resume is not implemented." >&2
        exit 3
    fi
done
"$PYTHON_BIN" "$CONTROL" --mode analyze --study "$STUDY_DIR" --cleanup-fork 2>&1 | tee "$STUDY_DIR/logs/analysis.log"
printf 'Completed. Curves, CSV and validation: %s/analysis\n' "$STUDY_DIR"
