#!/usr/bin/env bash
# Two PALU seeds -> proxy search -> held-out ES -> optional standard TOFU.
set -euo pipefail
PALU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_${MODEL}_full}"
MODEL_REVISION="${MODEL_REVISION:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEEDS="${SEEDS:-0 1}"
FORGET_SPLIT="${FORGET_SPLIT:-forget05}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain95}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout05}"
DATASET_PATH="${DATASET_PATH:-/mnt/sda/cr/LLM_unlearning/datset/TOFU}"
RETAIN_LOGS="${RETAIN_LOGS:-${PALU_ROOT}/saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-${PALU_ROOT}/models/gibberish-detector}"
SOUP_DEPTH="${SOUP_DEPTH:-3}"
PROXY_QUESTIONS="${PROXY_QUESTIONS:-64}"
PROXY_SEED="${PROXY_SEED:-20261013}"
FULL_EVAL="${FULL_EVAL:-1}"
# Diagnostic probes are optional and expensive; this experiment defaults them off.
DIAGNOSTICS="${DIAGNOSTICS:-0}"
SOUP_OUTPUT_ROOT="${SOUP_OUTPUT_ROOT:-${PALU_ROOT}/artifacts/palu_efficient_soup}"
STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
SOUP_RUN_DIR="${SOUP_RUN_DIR:-${SOUP_OUTPUT_ROOT}/${FORGET_SPLIT}/${STAMP}}"
case "$FULL_EVAL" in 0|1) ;; *) printf 'FULL_EVAL must be 0 or 1\n' >&2; exit 2 ;; esac
DRY_RUN=0
if [[ $# == 1 && "$1" == --dry-run ]]; then
  DRY_RUN=1
elif [[ $# != 0 ]]; then
  printf 'Usage: bash scripts/palu/run_efficient_soup.sh [--dry-run]\n' >&2; exit 2
fi
read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ ${#SEED_ARRAY[@]} != 2 || "${SEED_ARRAY[0]}" == "${SEED_ARRAY[1]}" ]]; then
  printf 'SEEDS must contain exactly two distinct integer seeds\n' >&2; exit 2
fi
if [[ "$DRY_RUN" == 0 && "$FULL_EVAL" == 1 && ! -f "$RETAIN_LOGS" ]]; then
  printf 'Missing Retain evaluation logs: %s\n' "$RETAIN_LOGS" >&2; exit 2
fi
# Resolve before invoking the training driver, which records absolute paths.
SOUP_RUN_DIR="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$SOUP_RUN_DIR")"
mkdir -p "$(dirname -- "$SOUP_RUN_DIR")"
mkdir "$SOUP_RUN_DIR"
export PYTHON_BIN MODEL MODEL_PATH MODEL_REVISION GPU_IDS SEEDS FORGET_SPLIT RETAIN_SPLIT HOLDOUT_SPLIT
export DATASET_PATH RETAIN_LOGS CLASSIFIER_MODEL DIAGNOSTICS
TRAIN=(bash "${PALU_ROOT}/scripts/palu/run.sh" --skip-eval)
[[ "$DRY_RUN" == 0 ]] || TRAIN+=(--dry-run)
KEEP_MODEL=1 OUTPUT_DIR="${SOUP_RUN_DIR}/candidates" "${TRAIN[@]}"
COMMAND=("$PYTHON_BIN" "${PALU_ROOT}/scripts/palu/efficient_soup.py"
  --original "$MODEL_PATH" --model "$MODEL" --dataset "$DATASET_PATH"
  --candidates "${SOUP_RUN_DIR}/candidates/seed_${SEED_ARRAY[0]}" "${SOUP_RUN_DIR}/candidates/seed_${SEED_ARRAY[1]}"
  --output "${SOUP_RUN_DIR}/soup_search" --forget-config "${FORGET_SPLIT}_perturbed"
  --depth "$SOUP_DEPTH" --proxy-questions "$PROXY_QUESTIONS" --split-seed "$PROXY_SEED"
  --forget-split "$FORGET_SPLIT" --holdout-split "$HOLDOUT_SPLIT")
[[ -z "$MODEL_REVISION" ]] || COMMAND+=(--revision "$MODEL_REVISION")
if [[ "$FULL_EVAL" == 1 ]]; then
  COMMAND+=(--full-eval --retain-logs "$RETAIN_LOGS" --classifier-model "$CLASSIFIER_MODEL")
fi
[[ "$DRY_RUN" == 0 ]] || COMMAND+=(--dry-run)
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" "${COMMAND[@]}"
