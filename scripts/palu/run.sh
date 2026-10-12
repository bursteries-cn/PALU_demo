#!/usr/bin/env bash
# Edit this parameter block, then: bash scripts/palu/run.sh
# Environment assignments override these defaults, e.g. LR=1e-5 SEEDS="0 1 2" bash .../run.sh
set -euo pipefail
PALU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Models/data: Retain model weights and FQ evaluation logs are different inputs.
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_${MODEL}_full}"
MODEL_REVISION="${MODEL_REVISION:-}"
FORGET_SPLIT="${FORGET_SPLIT:-forget05}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain95}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout05}"
RETAIN_MODEL_PATH="${RETAIN_MODEL_PATH:-${PALU_ROOT}/models/tofu_${MODEL}_${RETAIN_SPLIT}}"
RETAIN_MODEL_REVISION="${RETAIN_MODEL_REVISION:-}"
RETAIN_LOGS="${RETAIN_LOGS:-${PALU_ROOT}/saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
DATASET_PATH="${DATASET_PATH:-/mnt/sda/cr/LLM_unlearning/datset/TOFU}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-${PALU_ROOT}/models/gibberish-detector}"

# PALU training parameters, shared across seeds.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEEDS="${SEEDS:-0 1 2}"
LR="${LR:-2e-5}"
ALPHA="${ALPHA:-0.2}"
GAMMA="${GAMMA:-1.0}"
TOP_K="${TOP_K:-5000}"
FIRST_N="${FIRST_N:-3}"
EPOCHS="${EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:--1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-}" # Empty: choose effective batch 32 from GPU count.
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29513}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
KEEP_MODEL="${KEEP_MODEL:-0}" # 0: remove final weights after successful evaluation of each seed.

# Diagnostics before lm_head: Forget h[first sensitive - 1], Retain h[first answer - 1].
DIAGNOSTICS="${DIAGNOSTICS:-1}"
LOSS_HISTORY="${LOSS_HISTORY:-1}"
PROBE_EVERY="${PROBE_EVERY:-5}"
PROBE_FORGET_SAMPLES="${PROBE_FORGET_SAMPLES:-32}" # Positive integer or all.
PROBE_RETAIN_SAMPLES="${PROBE_RETAIN_SAMPLES:-32}"
PROBE_SEED="${PROBE_SEED:-20261011}"
PROBE_TEMPERATURE="${PROBE_TEMPERATURE:-1.0}"
# Saving: {baseline}/{experiment}/{time}/seed_<seed>; matching new seeds share time.
OUTPUT_ROOT="${OUTPUT_ROOT:-${PALU_ROOT}/artifacts/palu}"
EXPERIMENT="${EXPERIMENT:-${FORGET_SPLIT}_first_sensitive}"
NEW_GROUP="${NEW_GROUP:-0}" # 1: start a fresh timestamp, even when matching seeds can join.
OUTPUT_DIR="${OUTPUT_DIR:-}" # Optional legacy exact directory override.
for FLAG in "$KEEP_MODEL" "$LOSS_HISTORY" "$DIAGNOSTICS" "$NEW_GROUP"; do
  case "$FLAG" in 0|1) ;; *) printf 'KEEP_MODEL, LOSS_HISTORY, DIAGNOSTICS and NEW_GROUP must be 0 or 1\n' >&2; exit 2 ;; esac
done

read -r -a SEED_ARRAY <<< "$SEEDS"
COMMAND=("$PYTHON_BIN" "$PALU_ROOT/scripts/palu/run_experiment.py"
  --output-root "$OUTPUT_ROOT" --experiment "$EXPERIMENT" --gpu "$GPU_IDS" --seeds "${SEED_ARRAY[@]}"
  --model "$MODEL" --model-path "$MODEL_PATH" --retain-logs "$RETAIN_LOGS"
  --forget-split "$FORGET_SPLIT" --retain-split "$RETAIN_SPLIT" --holdout-split "$HOLDOUT_SPLIT"
  --lr "$LR" --alpha "$ALPHA" --gamma "$GAMMA" --top-k "$TOP_K" --first-n "$FIRST_N"
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BATCH_SIZE"
  --max-grad-norm "$MAX_GRAD_NORM" --main-process-port "$MAIN_PROCESS_PORT"
  --eval-batch-size "$EVAL_BATCH_SIZE" --eval-dataset "$DATASET_PATH" --classifier-model "$CLASSIFIER_MODEL"
  --override "data.retain.TOFU_QA_retain.args.hf_args.path=\"${DATASET_PATH}\"")
[[ -z "$GRAD_ACCUM" ]] || COMMAND+=(--grad-accum "$GRAD_ACCUM")
[[ -z "$OUTPUT_DIR" ]] || COMMAND+=(--output "$OUTPUT_DIR")
[[ "$NEW_GROUP" == 0 ]] || COMMAND+=(--new-group)
[[ -z "$MODEL_REVISION" ]] || COMMAND+=(--model-revision "$MODEL_REVISION")
[[ "$KEEP_MODEL" == 0 ]] || COMMAND+=(--keep-model)
[[ "$LOSS_HISTORY" == 1 ]] || COMMAND+=(--no-loss-history)
if [[ "$DIAGNOSTICS" == 1 ]]; then
  COMMAND+=(--retain-model-path "$RETAIN_MODEL_PATH" --probe-every "$PROBE_EVERY"
    --probe-forget-samples "$PROBE_FORGET_SAMPLES" --probe-retain-samples "$PROBE_RETAIN_SAMPLES"
    --probe-seed "$PROBE_SEED" --probe-temperature "$PROBE_TEMPERATURE")
  [[ -z "$RETAIN_MODEL_REVISION" ]] || COMMAND+=(--retain-model-revision "$RETAIN_MODEL_REVISION")
fi
export PYTHONUNBUFFERED=1
printf 'Baseline: %s; experiment: %s\nGPUs: %s; seeds: %s; LR: %s; alpha: %s; gamma: %s\n' \
  "$OUTPUT_ROOT" "$EXPERIMENT" "$GPU_IDS" "$SEEDS" "$LR" "$ALPHA" "$GAMMA"
exec "${COMMAND[@]}" "$@"
