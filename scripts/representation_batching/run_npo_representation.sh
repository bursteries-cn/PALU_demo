#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1"
MANIFEST_PATH=""
MODEL_PATH="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
MODEL_REVISION=""
DATASET_PATH="/mnt/sda/cr/LLM_unlearning/datset/TOFU"
DATASET_REVISION=""
OUTPUT_ROOT="./saves/unlearn/tofu/forget05/Llama-3.1-8B-Instruct/representation_npo"
EXACT_OUTPUT_DIR=""
LEARNING_RATE="2e-5"
SEED="0"
MAX_STEPS="-1"
DO_SAVE="true"

usage() {
    cat <<'EOF'
Usage:
  scripts/representation_batching/run_npo_representation.sh \
    --manifest artifacts/representation_batching/seed-0/R.jsonl [options]

Options:
  --manifest PATH       Required R/S/D/P manifest.
  --gpu IDS             Exactly two comma-separated GPU ids (default: 0,1).
  --model PATH_OR_ID    Full TOFU LLaMA-3.1-8B checkpoint.
  --model-revision REV  Optional immutable model revision.
  --dataset PATH_OR_ID  TOFU dataset path or Hub id.
  --dataset-revision REV Optional immutable dataset revision.
  --output-root PATH    Parent output directory.
  --output-dir PATH     Exact NEW run directory (used by run_seed.sh).
  --lr VALUE            Learning rate (default: 2e-5).
  --seed INT            Training seed; should match the manifest seed.
  --max-steps INT       Stop after this many optimizer steps; use 2 for smoke.
  --no-save             Run without saving the final model (smoke tests only).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST_PATH="$2"; shift 2 ;;
        --gpu) GPU_IDS="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --model-revision) MODEL_REVISION="$2"; shift 2 ;;
        --dataset) DATASET_PATH="$2"; shift 2 ;;
        --dataset-revision) DATASET_REVISION="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --output-dir) EXACT_OUTPUT_DIR="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --no-save) DO_SAVE="false"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${MANIFEST_PATH}" || ! -f "${MANIFEST_PATH}" ]]; then
    echo "--manifest must point to an existing JSONL manifest" >&2
    exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 2 ]]; then
    echo "This experiment manifest expects exactly two GPUs; got: ${GPU_IDS}" >&2
    exit 2
fi

ARM_NAME=$(basename "${MANIFEST_PATH}" .jsonl)
RUN_STAMP=$(date "+%Y%m%d-%H%M%S")
TASK_NAME="representation_npo_${ARM_NAME}_seed${SEED}_${RUN_STAMP}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK_NAME}"
if [[ -n "${EXACT_OUTPUT_DIR}" ]]; then
    if [[ -e "${EXACT_OUTPUT_DIR}" ]]; then
        echo "--output-dir must be new; refusing to overwrite ${EXACT_OUTPUT_DIR}" >&2
        exit 2
    fi
    OUTPUT_DIR="${EXACT_OUTPUT_DIR}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export WANDB_PROJECT="${WANDB_PROJECT:-npo-representation-batching}"
export WANDB_MODE="${WANDB_MODE:-offline}"

COMMAND=(
    accelerate launch
    --config_file configs/accelerate/default_config.yaml
    --num_processes 2
    src/train.py --config-name=unlearn.yaml
    experiment=unlearn/tofu/representation_npo
    "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}"
    "model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH}"
    "data.forget.TOFU_QA_forget.args.hf_args.path=${DATASET_PATH}"
    "data.retain.TOFU_QA_retain.args.hf_args.path=${DATASET_PATH}"
    "trainer.method_args.batch_manifest_path=${MANIFEST_PATH}"
    "trainer.args.learning_rate=${LEARNING_RATE}"
    "trainer.args.seed=${SEED}"
    "trainer.args.max_steps=${MAX_STEPS}"
    "trainer.args.run_name=${TASK_NAME}"
    "paths.output_dir=${OUTPUT_DIR}"
    "task_name=${TASK_NAME}"
    "do_save=${DO_SAVE}"
)
if [[ -n "${MODEL_REVISION}" ]]; then
    COMMAND+=(
        "+model.model_args.revision=${MODEL_REVISION}"
        "+model.tokenizer_args.revision=${MODEL_REVISION}"
    )
fi
if [[ -n "${DATASET_REVISION}" ]]; then
    COMMAND+=(
        "+data.forget.TOFU_QA_forget.args.hf_args.revision=${DATASET_REVISION}"
        "+data.retain.TOFU_QA_retain.args.hf_args.revision=${DATASET_REVISION}"
    )
fi

"${COMMAND[@]}"
if [[ "${DO_SAVE}" == "true" ]]; then
    # Written only after both distributed training and final model saving return.
    printf '%s\n' '{"status":"completed"}' > "${OUTPUT_DIR}/model_save_complete.json"
fi
