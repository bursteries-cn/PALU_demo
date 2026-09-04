#!/bin/bash
set -e

DATE=$(date "+%m%d")
TIME=$(date "+%H%M%S")
# 结果目录里的日期层，整个 sweep 共用脚本启动时的日期
DATE_DIR=$(date "+%Y-%m-%d")


GPU="0,"
MODEL="Llama-3.1-8B-Instruct"

REPORTTO="wandb"
WANDB_PROJECT="anonymous_code_unlearning"
DO_SAVE="false"

TRAINER="NPO"
PRETRAINED_PATH="open-unlearning/tofu_Llama-3.1-8B-Instruct_full"

splits=(
    "forget05 holdout05 retain95"
)
# lr, batchsize, grad_acc, epochs
# 以下为默认值，可通过命令行参数覆盖
lr_set=("2e-5")
bz_set=("8 4")
beta_set=(0.1)
alpha_set=(1.0)
gamma_set=(1.0)
epoch_set=(10)

usage() {
    cat <<EOF
用法: $0 [选项]

  --lr       "<v1 v2 ...>"   学习率           (默认: ${lr_set[*]})
  --beta     "<v1 v2 ...>"   NPO beta         (默认: ${beta_set[*]})
  --alpha    "<v1 v2 ...>"   retain 损失权重  (默认: ${alpha_set[*]})
  --gamma    "<v1 v2 ...>"   forget 损失权重  (默认: ${gamma_set[*]})
  --gpu      <ids>           CUDA_VISIBLE_DEVICES (默认: ${GPU})
  -h, --help                 显示本帮助

除 --gpu 外，每个选项都可以传入用空格分隔的多个值来做 sweep，例如:
  $0 --gpu 0,1 --lr "1e-5 5e-5" --beta "0.05 0.1" --alpha 1.0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr)       read -r -a lr_set    <<< "$2"; shift 2 ;;
        --beta)     read -r -a beta_set  <<< "$2"; shift 2 ;;
        --alpha)    read -r -a alpha_set <<< "$2"; shift 2 ;;
        --gamma)    read -r -a gamma_set <<< "$2"; shift 2 ;;
        --lr=*)     read -r -a lr_set    <<< "${1#*=}"; shift ;;
        --beta=*)   read -r -a beta_set  <<< "${1#*=}"; shift ;;
        --alpha=*)  read -r -a alpha_set <<< "${1#*=}"; shift ;;
        --gamma=*)  read -r -a gamma_set <<< "${1#*=}"; shift ;;
        --gpu)      GPU="$2"; shift 2 ;;
        --gpu=*)    GPU="${1#*=}"; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "lr_set    = ${lr_set[*]}"
echo "beta_set  = ${beta_set[*]}"
echo "alpha_set = ${alpha_set[*]}"
echo "gamma_set = ${gamma_set[*]}"

for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for epochs in "${epoch_set[@]}"; do
                for beta in "${beta_set[@]}"; do
                    for alpha in "${alpha_set[@]}"; do
                        for gamma in "${gamma_set[@]}"; do
                            # Args ========================================
                            forget_split=$(echo $split | cut -d' ' -f1)
                            holdout_split=$(echo $split | cut -d' ' -f2)
                            retain_split=$(echo $split | cut -d' ' -f3)

                            bsz=$(echo $bz | cut -d' ' -f1)
                            grad_acc=$(echo $bz | cut -d' ' -f2)

                            # learning_rate, batchsize, grad_acc, epochs
                            SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_a${alpha}_g${gamma}_beta${beta}_e${epochs}_day${DATE}_time${TIME}"
                            TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
                            OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${DATE_DIR}/${TRAINER}/${SUFFIX}"

                            # TRAIN COMMAND =================================
                            export WANDB_PROJECT=${WANDB_PROJECT}
                            python src/train.py --config-name=unlearn.yaml \
                                experiment=unlearn/tofu/default \
                                trainer=${TRAINER} \
                                model=${MODEL} \
                                model.model_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
                                model.tokenizer_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
                                forget_split=${forget_split} \
                                holdout_split=${holdout_split} \
                                retain_split=${retain_split} \
                                task_name=${TASK_NAME} \
                                paths.output_dir="${OUTPUT_DIR}" \
                                do_save=${DO_SAVE} \
                                eval.tofu.retain_logs_path=./saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json \
                                trainer.args.ddp_find_unused_parameters=true \
                                trainer.args.gradient_checkpointing=true \
                                trainer.args.report_to=${REPORTTO} \
                                trainer.args.run_name=${TASK_NAME} \
                                trainer.args.logging_steps=1 \
                                trainer.args.learning_rate=$lr \
                                trainer.args.per_device_train_batch_size=$bsz \
                                trainer.args.gradient_accumulation_steps=$grad_acc \
                                trainer.args.num_train_epochs=$epochs \
                                trainer.args.eval_strategy=epoch \
                                trainer.args.eval_on_start=True \
                                trainer.method_args.gamma=$gamma \
                                trainer.method_args.alpha=$alpha \
                                trainer.method_args.beta=$beta \
                                trainer.method_args.retain_loss_type=NLL
                        done
                    done
                done
            done
        done
    done
done
