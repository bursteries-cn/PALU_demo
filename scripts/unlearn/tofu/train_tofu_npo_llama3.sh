#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${PROJECT_ROOT}"

# Local NPO baseline: four GPUs, ordinary random sampling, existing SDPA config.
GPU="0,1,2,3"
MODEL="Llama-3.1-8B-Instruct"
PRETRAINED_PATH="open-unlearning/tofu_${MODEL}_full"
DATASET_PATH="/mnt/sda/cr/LLM_unlearning/datset/TOFU"
RETAIN_LOGS="${PROJECT_ROOT}/saves/eval/tofu_${MODEL}_retain95/TOFU_EVAL.json"
OUTPUT_ROOT="${PROJECT_ROOT}/saves/unlearn/tofu/forget05/${MODEL}/npo_baseline"
EXACT_OUTPUT_DIR=""
PYTHON_BIN="${NPO_PYTHON:-python3}"
SEED=0
EPOCHS=10
PORT=29501
MAX_GRAD_NORM=0
REPORTTO="none"
CLASSIFIER_MODEL=""
KEEP_MODEL=false
DRY_RUN=false
lr_set=("1e-5")
beta_set=("0.1")
alpha_set=("1.0")
gamma_set=("1.0")

usage() {
    cat <<'EOF'
Usage: bash scripts/unlearn/tofu/train_tofu_npo_llama3.sh [options]

Default: local NPO, forget05/retain95, GPUs 0,1,2,3, per-device batch 4,
         accumulation 2 (effective batch 32), lr 1e-5, 10 epochs,
         one warmup epoch, linear scheduler, paged_adamw_32bit, clipping off.
         Uses the existing attention configuration; no FlashAttention install/check.

  --gpu IDS             Exactly four distinct GPU IDs (default: 0,1,2,3).
  --seed INT            Training and evaluation seed (default: 0).
  --epochs INT          Training epochs (default: 10).
  --lr "VALUES"         Learning-rate sweep (default: 1e-5).
  --beta "VALUES"       NPO beta sweep (default: 0.1).
  --alpha "VALUES"      Retain coefficient sweep (default: 1.0).
  --gamma "VALUES"      Forget coefficient sweep (default: 1.0).
  --max-grad-norm VALUE Gradient clipping threshold; 0 disables it (default: 0).
  --model PATH_OR_ID    Full Llama-3.1-8B checkpoint.
  --dataset PATH_OR_ID  TOFU dataset, for both training and evaluation.
  --retain-logs PATH    Matched Retain95 TOFU_EVAL.json.
  --classifier-model P Gibberish detector path/Hub ID; otherwise uses current config.
  --output-root PATH    Parent of automatically named run directories.
  --output-dir PATH     Exact NEW directory; single configuration only.
  --python PATH         Training-environment Python (default: python3).
  --port INT            Distributed port (default: 29501).
  --report-to NAME      none, wandb or tensorboard (default: none).
  --keep-model          Keep final weights after successful evaluation.
  --no-save            Delete weights after successful evaluation (the default).
  --dry-run            Print commands without loading models or writing results.
  -h, --help            Show this help.

Training temporarily saves weights for single-GPU final evaluation. On failure,
weights and logs are preserved. Evaluation uses the first GPU in --gpu.
EOF
}

die() { echo "Error: $*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "Missing value for $1"; }
absolute_path() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;; esac; }
print_command() { "${PYTHON_BIN}" -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@"; }

while [[ $# -gt 0 ]]; do
    case "$1" in --*=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;; esac
    case "$1" in
        --lr) need_value "$@"; read -r -a lr_set <<< "$2"; shift 2 ;;
        --beta) need_value "$@"; read -r -a beta_set <<< "$2"; shift 2 ;;
        --alpha) need_value "$@"; read -r -a alpha_set <<< "$2"; shift 2 ;;
        --gamma) need_value "$@"; read -r -a gamma_set <<< "$2"; shift 2 ;;
        --gpu) need_value "$@"; GPU="$2"; shift 2 ;;
        --seed) need_value "$@"; SEED="$2"; shift 2 ;;
        --epochs) need_value "$@"; EPOCHS="$2"; shift 2 ;;
        --max-grad-norm) need_value "$@"; MAX_GRAD_NORM="$2"; shift 2 ;;
        --model) need_value "$@"; PRETRAINED_PATH="$2"; shift 2 ;;
        --dataset) need_value "$@"; DATASET_PATH="$2"; shift 2 ;;
        --retain-logs) need_value "$@"; RETAIN_LOGS=$(absolute_path "$2"); shift 2 ;;
        --classifier-model) need_value "$@"; CLASSIFIER_MODEL="$2"; shift 2 ;;
        --output-root) need_value "$@"; OUTPUT_ROOT=$(absolute_path "$2"); shift 2 ;;
        --output-dir) need_value "$@"; EXACT_OUTPUT_DIR=$(absolute_path "$2"); shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --port|--main-process-port) need_value "$@"; PORT="$2"; shift 2 ;;
        --report-to) need_value "$@"; REPORTTO="$2"; shift 2 ;;
        --keep-model) KEEP_MODEL=true; shift ;;
        --no-save|--discard-model-after-eval) KEEP_MODEL=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ "${GPU}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || die "--gpu requires four IDs, e.g. 0,1,2,3"
IFS=',' read -r -a GPU_IDS <<< "${GPU}"
for ((i=0; i<4; i++)); do
    for ((j=0; j<i; j++)); do
        [[ "${GPU_IDS[i]}" != "${GPU_IDS[j]}" ]] || die "GPU IDs must be distinct"
    done
done
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "--seed must be a nonnegative integer"
[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer"
[[ "${PORT}" =~ ^[1-9][0-9]*$ && ${#PORT} -le 5 ]] || die "Invalid port"
(( PORT <= 65535 )) || die "Invalid port"
case "${REPORTTO}" in none|wandb|tensorboard) ;; *) die "Invalid --report-to" ;; esac
TOTAL=$((${#lr_set[@]} * ${#beta_set[@]} * ${#alpha_set[@]} * ${#gamma_set[@]}))
(( TOTAL > 0 )) || die "Sweep lists must not be empty"
[[ -z "${EXACT_OUTPUT_DIR}" || ${TOTAL} -eq 1 ]] || die "--output-dir supports one configuration; use --output-root for a sweep"

"${PYTHON_BIN}" - "${MAX_GRAD_NORM}" "${lr_set[*]}" "${beta_set[*]}" "${alpha_set[*]}" "${gamma_set[*]}" <<'PY'
import math, sys
for name, text, positive in zip(('max-grad-norm', 'lr', 'beta', 'alpha', 'gamma'), sys.argv[1:], (False, True, True, False, False)):
    if not text.split():
        raise SystemExit(f'Empty --{name}')
    for item in text.split():
        try:
            value = float(item)
        except ValueError:
            raise SystemExit(f'Invalid --{name}: {item}')
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise SystemExit(f'Invalid --{name}: {item}')
PY

if [[ "${DRY_RUN}" == false ]]; then
    [[ -z "${EXACT_OUTPUT_DIR}" || ! -e "${EXACT_OUTPUT_DIR}" ]] || die "Output already exists: ${EXACT_OUTPUT_DIR}"
    [[ -f "${RETAIN_LOGS}" ]] || die "Retain95 log not found: ${RETAIN_LOGS}; use --retain-logs"
    "${PYTHON_BIN}" - "${RETAIN_LOGS}" <<'PY'
import json, pathlib, sys
reference = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not isinstance(reference.get('forget_truth_ratio'), dict):
    raise SystemExit('Retain95 log must contain forget_truth_ratio; do not use TOFU_SUMMARY.json.')
PY
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.device_count() == 4, "Four visible CUDA GPUs are required"'
fi

export WANDB_PROJECT="${WANDB_PROJECT:-npo-baseline}"
export WANDB_MODE="${WANDB_MODE:-offline}"
RUN_STAMP=$(date "+%Y%m%d-%H%M%S")
for lr in "${lr_set[@]}"; do
    for beta in "${beta_set[@]}"; do
        for alpha in "${alpha_set[@]}"; do
            for gamma in "${gamma_set[@]}"; do
                SUFFIX="seed${SEED}_lr${lr}_beta${beta}_a${alpha}_g${gamma}_e${EPOCHS}_${RUN_STAMP}_$$"
                OUTPUT_DIR="${EXACT_OUTPUT_DIR:-${OUTPUT_ROOT}/${SUFFIX}}"
                TASK_NAME="npo_${MODEL}_forget05_${SUFFIX}"
                [[ ! -e "${OUTPUT_DIR}" ]] || die "Output already exists: ${OUTPUT_DIR}"
                TRAIN_COMMAND=(
                    "${PYTHON_BIN}" -m accelerate.commands.launch
                    --config_file configs/accelerate/default_config.yaml
                    --num_processes 4 --main_process_port "${PORT}"
                    src/train.py --config-name=unlearn.yaml
                    experiment=unlearn/tofu/default trainer=NPO "model=${MODEL}"
                    "model.model_args.pretrained_model_name_or_path=${PRETRAINED_PATH}"
                    "model.tokenizer_args.pretrained_model_name_or_path=${PRETRAINED_PATH}"
                    forget_split=forget05 holdout_split=holdout05 retain_split=retain95
                    "data.forget.TOFU_QA_forget.args.hf_args.path=${DATASET_PATH}"
                    "data.retain.TOFU_QA_retain.args.hf_args.path=${DATASET_PATH}"
                    "paths.output_dir=${OUTPUT_DIR}" "task_name=${TASK_NAME}"
                    do_save=true '~eval.tofu'
                    trainer.args.per_device_train_batch_size=4
                    trainer.args.gradient_accumulation_steps=2
                    "trainer.args.learning_rate=${lr}" "trainer.args.num_train_epochs=${EPOCHS}"
                    "trainer.args.seed=${SEED}" trainer.args.warmup_epochs=1.0
                    ++trainer.args.lr_scheduler_type=linear trainer.args.optim=paged_adamw_32bit
                    trainer.args.weight_decay=0.01 "++trainer.args.max_grad_norm=${MAX_GRAD_NORM}"
                    trainer.args.ddp_find_unused_parameters=true trainer.args.gradient_checkpointing=true
                    trainer.args.do_eval=false trainer.args.eval_strategy=no trainer.args.eval_on_start=false
                    trainer.args.save_strategy=no trainer.args.logging_steps=1
                    "trainer.args.report_to=${REPORTTO}" "trainer.args.run_name=${TASK_NAME}"
                    "trainer.method_args.beta=${beta}" "trainer.method_args.alpha=${alpha}"
                    "trainer.method_args.gamma=${gamma}" trainer.method_args.retain_loss_type=NLL
                )
                EVAL_COMMAND=(
                    "${PYTHON_BIN}" src/eval.py experiment=eval/tofu/default "model=${MODEL}"
                    "model.model_args.pretrained_model_name_or_path=${OUTPUT_DIR}"
                    "model.tokenizer_args.pretrained_model_name_or_path=${OUTPUT_DIR}"
                    forget_split=forget05 holdout_split=holdout05 "seed=${SEED}"
                    "retain_logs_path=${RETAIN_LOGS}" "+tofu_dataset_path=${DATASET_PATH}"
                    "paths.output_dir=${OUTPUT_DIR}/evals" "task_name=${TASK_NAME}_final_eval"
                    eval.tofu.overwrite=true
                )
                if [[ -n "${CLASSIFIER_MODEL}" ]]; then
                    EVAL_COMMAND+=(
                        "eval.tofu.metrics.forget_Q_A_gibberish.classifier_model_args.pretrained_model_name_or_path=${CLASSIFIER_MODEL}"
                        "eval.tofu.metrics.forget_Q_A_gibberish.classifier_tokenization_args.pretrained_model_name_or_path=${CLASSIFIER_MODEL}"
                    )
                fi
                echo "Output: ${OUTPUT_DIR}; effective batch: 4 GPUs x 4 x 2 = 32"
                print_command env "CUDA_VISIBLE_DEVICES=${GPU}" "${TRAIN_COMMAND[@]}"
                print_command env "CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}" "${EVAL_COMMAND[@]}"
                if [[ "${DRY_RUN}" == true ]]; then continue; fi
                mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
                mkdir -- "${OUTPUT_DIR}"
                {
                    print_command cd "${PROJECT_ROOT}"
                    print_command env "CUDA_VISIBLE_DEVICES=${GPU}" "${TRAIN_COMMAND[@]}"
                    print_command env "CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}" "${EVAL_COMMAND[@]}"
                } > "${OUTPUT_DIR}/launch_commands.sh"
                CUDA_VISIBLE_DEVICES="${GPU}" "${TRAIN_COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
                "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
state = json.loads((run / 'trainer_state.json').read_text())
if state.get('max_steps', 0) <= 0 or state.get('global_step') != state['max_steps']:
    raise SystemExit('Training did not reach its planned optimizer-step count; weights preserved.')
PY
                CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${EVAL_COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
                "${PYTHON_BIN}" - "${SCRIPT_DIR}" "${OUTPUT_DIR}" "${KEEP_MODEL}" <<'PY'
import json, math, pathlib, sys
sys.path.insert(0, sys.argv[1])
# Shared stdlib-only artifact checks; does not launch/check the upstream environment.
from run_official_npo import check_evaluation, remove_weights, write_json
run = pathlib.Path(sys.argv[2]).resolve()
scores = check_evaluation(run)
for key in ('exact_memorization', 'forget_Q_A_gibberish'):
    value = scores.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f'Missing/non-finite {key}; weights preserved.')
provenance = json.loads((run / 'evals/evaluation_provenance.json').read_text())
evaluated_model = pathlib.Path(provenance['config']['model']['model_args']['pretrained_model_name_or_path']).resolve()
if provenance.get('status') != 'completed' or evaluated_model != run:
    raise SystemExit('Final evaluation does not match this run; weights preserved.')
removed = [] if sys.argv[3] == 'true' else remove_weights(run)
write_json(run / 'npo_run_complete.json', {
    'status': 'completed', 'metrics': scores, 'removed_weight_files': removed,
    'keep_model_weights': sys.argv[3] == 'true',
})
print(f"Completed: {run / 'evals/TOFU_SUMMARY.json'}")
PY
            done
        done
    done
done
