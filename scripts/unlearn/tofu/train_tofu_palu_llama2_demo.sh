#!/bin/bash
set -e

DATE=$(date "+%m%d")
TIME=$(date "+%H%M%S")
# 结果目录里的日期层，整个 sweep 共用脚本启动时的日期
DATE_DIR=$(date "+%Y-%m-%d")


GPU="2,"
MODEL="Llama-2-7b-chat-hf" 

REPORTTO="wandb"
WANDB_PROJECT="anonymous_code_unlearning"
DO_SAVE="true"

TRAINER="PALU"
PRETRAINED_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_full"

splits=(
    "forget05 holdout05 retain95"
    # "forget01 holdout01 retain99"
    # "forget10 holdout10 retain90"
)
# lr, batchsize, grad_acc, epochs
# 以下为默认值，可通过命令行参数覆盖
lr_set=("5e-5")
bz_set=("8 4")
target_mode_set=("mean")
alpha_set=(0.2)
topk_set=(5000)
first_n_set=(2 3 5)
epoch_set=(10)

usage() {
    cat <<EOF
用法: $0 [选项]

  --lr       "<v1 v2 ...>"   学习率           (默认: ${lr_set[*]})
  --alpha    "<v1 v2 ...>"   alpha            (默认: ${alpha_set[*]})
  --topk     "<v1 v2 ...>"   top_k            (默认: ${topk_set[*]})
  --first_n  "<v1 v2 ...>"   first_n          (默认: ${first_n_set[*]})
  --gpu      <ids>           CUDA_VISIBLE_DEVICES (默认: ${GPU})
  -h, --help                 显示本帮助

除 --gpu 外，每个选项都可以传入用空格分隔的多个值来做 sweep，例如:
  $0 --gpu 0,1 --lr "1e-5 5e-5" --alpha 0.2 --topk 5000 --first_n "2 3 5"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr)       read -r -a lr_set      <<< "$2"; shift 2 ;;
        --alpha)    read -r -a alpha_set   <<< "$2"; shift 2 ;;
        --topk)     read -r -a topk_set    <<< "$2"; shift 2 ;;
        --first_n)  read -r -a first_n_set <<< "$2"; shift 2 ;;
        --lr=*)      read -r -a lr_set      <<< "${1#*=}"; shift ;;
        --alpha=*)   read -r -a alpha_set   <<< "${1#*=}"; shift ;;
        --topk=*)    read -r -a topk_set    <<< "${1#*=}"; shift ;;
        --first_n=*) read -r -a first_n_set <<< "${1#*=}"; shift ;;
        --gpu)      GPU="$2"; shift 2 ;;
        --gpu=*)    GPU="${1#*=}"; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "lr_set      = ${lr_set[*]}"
echo "alpha_set   = ${alpha_set[*]}"
echo "topk_set    = ${topk_set[*]}"
echo "first_n_set = ${first_n_set[*]}"

for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for target_mode in "${target_mode_set[@]}"; do
                for epochs in "${epoch_set[@]}"; do
                    for alpha in "${alpha_set[@]}"; do
                        for topk in "${topk_set[@]}"; do
                            for first_n in "${first_n_set[@]}"; do
                                # Args ========================================
                                forget_split=$(echo $split | cut -d' ' -f1)
                                holdout_split=$(echo $split | cut -d' ' -f2)
                                retain_split=$(echo $split | cut -d' ' -f3)

                                bsz=$(echo $bz | cut -d' ' -f1)
                                grad_acc=$(echo $bz | cut -d' ' -f2)

                                # learning_rate, batchsize, grad_acc, epochs
                                SUFFIX="target_mode${target_mode}_first_n${first_n}_lr${lr}_b${bsz}_ga${grad_acc}_a${alpha}_topk${topk}_e${epochs}_day${DATE}_time${TIME}"
                                TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
                                OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${DATE_DIR}/${TRAINER}_acl/${SUFFIX}"

                                # TRAIN COMMAND =================================
                                export WANDB_PROJECT=${WANDB_PROJECT}
                                python src/train.py --config-name=unlearn.yaml \
                                    experiment=unlearn/tofu/tpo \
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
                                    trainer.method_args.gamma=1.0 \
                                    trainer.method_args.alpha=$alpha \
                                    trainer.method_args.retain_loss_type=NLL \
                                    trainer.method_args.top_k=$topk \
                                    trainer.method_args.target_mode=${target_mode} \
                                    trainer.method_args.first_n=${first_n}
                            done
                        done
                    done
                done
            done
        done
    done
done