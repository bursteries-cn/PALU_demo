#!/usr/bin/env bash
# Read one saved checkpoint; run each backend once in a fresh four-rank process.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
: "${STUDY_DIR:?Set STUDY_DIR to the source frozen branch study}"
: "${CHECKPOINT:?Set CHECKPOINT to a retained full model checkpoint}"
: "${BENCHMARK_DIR:?Set BENCHMARK_DIR to a new output directory}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$REPO_ROOT"
mkdir -p -- "$BENCHMARK_DIR"
EXTRA=()
if [[ "${SMOKE:-0}" == 1 ]]; then EXTRA+=(--smoke); fi
for BACKEND in legacy sharded; do
    if [[ -e "$BENCHMARK_DIR/$BACKEND" ]]; then
        echo "Output already exists: $BENCHMARK_DIR/$BACKEND. Preserve it and choose a fresh BENCHMARK_DIR." >&2
        exit 2
    fi
    "$PYTHON_BIN" scripts/palu/seed103_branch_experiment.py --mode check-gpus \
        --gpus "$FOUR_GPUS" --min-free-gib "${MIN_FREE_GIB:-40}"
    CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=4 \
        "$SCRIPT_DIR/benchmark_sharded_evaluation.py" --backend "$BACKEND" \
        --study "$STUDY_DIR" --checkpoint "$CHECKPOINT" --output "$BENCHMARK_DIR" \
        "${EXTRA[@]}" 2>&1 | tee "$BENCHMARK_DIR/${BACKEND}.log"
done
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_sharded_evaluation.py" \
    --backend compare --output "$BENCHMARK_DIR" | tee "$BENCHMARK_DIR/comparison.log"
