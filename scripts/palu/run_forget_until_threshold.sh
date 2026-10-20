#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_STUDY="${SOURCE_STUDY:-$REPO_ROOT/artifacts/token_equal_seed103/20261017_192348}"
STUDY_DIR="${STUDY_DIR:-$REPO_ROOT/artifacts/forget_threshold_light/$(date +%Y%m%d_%H%M%S)}"
PROBE_INTERVAL="${PROBE_INTERVAL:-epoch}"
FOUR_GPUS="${FOUR_GPUS:-0,1,2,3}"
MODE="${MODE:-run}"
read -r -a SEED_LIST <<< "${SEEDS:-103 104 105 106}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$REPO_ROOT"
CONTROL="$SCRIPT_DIR/forget_until_threshold.py"
if [[ "$MODE" != run && "$MODE" != dry-run ]]; then
  printf 'MODE must be run or dry-run\n' >&2; exit 2
fi
if [[ ${#SEED_LIST[@]} == 0 ]]; then printf 'SEEDS must not be empty\n' >&2; exit 2; fi
# Fail before preparing any seed if the list is malformed or repeats an output.
SEEN=" "
for seed in "${SEED_LIST[@]}"; do
  if [[ ! "$seed" =~ ^(0|[1-9][0-9]*)$ || "$SEEN" == *" $seed "* ]]; then
    printf 'SEEDS must contain distinct nonnegative integer seeds\n' >&2; exit 2
  fi
  SEEN+="$seed "
done
if [[ "$MODE" == run && -e "$STUDY_DIR" ]]; then
  printf 'Choose a fresh STUDY_DIR; refusing to overwrite %s\n' "$STUDY_DIR" >&2; exit 2
fi
printf 'Sequential seeds: %s; GPUs: %s; threshold probes: %s; no retain stage will run.\n' "${SEED_LIST[*]}" "$FOUR_GPUS" "$PROBE_INTERVAL"
printf 'Evaluation: F200 F+KL each probe; FQ/TR initially and every 2 epochs; MU once at Forget stop.\n'
"$PYTHON_BIN" -S - "${SEED_LIST[@]}" <<'PY'
import sys
seeds = [int(value) for value in sys.argv[1:]]
if any(not 0 <= seed < 2**32 for seed in seeds):
    raise SystemExit("Every seed must be in [0, 2**32)")
count = len(seeds)
one = 128_000_000_000 / (1 << 30)
print(f"Approximate persistent model + Adam restart: {one:.1f} GiB/seed, {count*one:.1f} GiB total.")
print("Each preparation checks actual weight sizes and requires at least 160 GiB free; earlier seeds remain saved.")
PY
for seed in "${SEED_LIST[@]}"; do
  SEED_STUDY="$STUDY_DIR/seed_$seed"
  ARGS=(--study "$SEED_STUDY" --source-study "$SOURCE_STUDY" --seed "$seed"
    --threshold-low "${THRESHOLD_LOW:-0.4}" --threshold-high "${THRESHOLD_HIGH:-0.5}"
    --max-updates "${MAX_UPDATES:-63}" --learning-rate "${LEARNING_RATE:-1e-5}"
    --probe-interval "$PROBE_INTERVAL")
  if [[ -n "${THRESHOLD_COMPARISON:-}" ]]; then
    ARGS+=(--threshold-comparison "$THRESHOLD_COMPARISON")
  fi
  if [[ "$MODE" == dry-run ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode dry-run "${ARGS[@]}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode prepare "${ARGS[@]}"
  mkdir -p "$SEED_STUDY/logs"
  trap 'rc=$?; printf "Stopped (exit %s); measurements preserved: %s\n" "$rc" "$SEED_STUDY" >&2; exit "$rc"' ERR
  "$PYTHON_BIN" "$SCRIPT_DIR/seed103_branch_experiment.py" --mode check-gpus \
    --gpus "$FOUR_GPUS" --min-free-gib "${MIN_FREE_GIB:-40}" > "$SEED_STUDY/logs/gpu_before.json"
  CUDA_VISIBLE_DEVICES="$FOUR_GPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 "$CONTROL" --mode train --study "$SEED_STUDY" \
    2>&1 | tee "$SEED_STUDY/logs/train.log"
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$CONTROL" --mode analyze --study "$SEED_STUDY" \
    2>&1 | tee "$SEED_STUDY/logs/analysis.log"
  printf 'Forget stage complete; checkpoint retained, no retain training: %s\n' "$SEED_STUDY"
done
if [[ "$MODE" == dry-run ]]; then
  printf 'Dry-run finished; no training started or result directory created: %s\n' "$STUDY_DIR"
else
  printf 'All requested seeds finished: %s\n' "$STUDY_DIR"
fi
