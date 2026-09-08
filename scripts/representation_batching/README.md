# NPO representation-guided effective batches

This directory implements the experiment in which cached representations from
the initial LLaMA-3.1-8B TOFU Full model determine the composition of NPO
effective batches. It does not reorder examples inside one microbatch and does
not refresh representations during unlearning.

The four first-stage arms are:

- `R`: random grouping without replacement.
- `S`: greedily place similar examples in one effective batch.
- `D`: greedily place dissimilar examples in one effective batch.
- `P`: permute the sample-to-feature mapping, then run the `S` algorithm.

Every arm covers every forget row exactly once per epoch. The global effective
batch is 20: two ranks, per-device microbatch 1, and ten accumulation steps.
Every epoch must divide into complete accumulation windows; samples are never
dropped, duplicated, or carried across an epoch boundary. Retain row ids are
generated once per seed and are identical at corresponding optimizer steps
across arms.

## 1. Server environment

Use the repository's pinned environment (`transformers==4.45.1`,
`accelerate==0.34.2`, `deepspeed==0.15.4`, `torch==2.4.1`). Accept the LLaMA
license and authenticate on the server with `hf auth login` or a secret
`HF_TOKEN`. Do not put a token in a script or config file.

For an offline server, copy the complete TOFU repository directory and the Full
model snapshot to local storage. Passing the absolute TOFU directory through
`--dataset` selects `<config>.json` directly, so the copied Dataset Card does
not need to expose Hub `BuilderConfig` metadata. Pass the local model directory
through `--model`, and omit revision arguments for both local paths.

All commands below run from the repository root. The local laptop is suitable
for manifest tests but is not expected to load the model.

## 2. Extract features once

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/representation_batching/extract_tofu_features.py \
  --model open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  --dataset locuslab/TOFU \
  --dataset-config forget05 \
  --batch-size 2 \
  --block two_thirds \
  --output-dir artifacts/representation_batching/features-forget05
```

The extractor uses the same LLaMA chat template assumptions as the training
configuration. It runs the complete question-answer sequence once: causal
masking lets it read question positions without seeing future answer tokens,
and the same forward pass supplies initial answer NLL. It stores only six pooled
vectors per row, not all token activations. `--max-length` is a guard: the
extractor fails instead of truncating tokens that the repository training path
would keep. Pass immutable `--model-revision` and `--dataset-revision` values
for a final run; the resolved model and tokenizer commit ids are recorded.

For LLaMA-3.1-8B (32 decoder blocks), `two_thirds` resolves to one-indexed block
22. The primary first-stage key is:

```text
block_22_question_last
```

Before continuing, inspect `feature_manifest.json`. The script fails if the
rendered-template tokenization disagrees with `apply_chat_template`, if the
question or answer span is absent, or if a sequence exceeds the configured
length guard.

## 3. Build manifests

First confirm the actual retain95 row count on the server. For standard TOFU it
is expected to be 3800, but pass the observed value rather than trusting this
note. Build one manifest directory per paired seed:

```bash
for seed in 0 1 2; do
  python scripts/representation_batching/build_batch_manifests.py \
    --features artifacts/representation_batching/features-forget05/features.npz \
    --feature-key block_22_question_last \
    --output-dir artifacts/representation_batching/seed-${seed} \
    --retain-size 3800 \
    --effective-batch-size 20 \
    --world-size 2 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 10 \
    --num-epochs 3 \
    --seed "${seed}"
done
```

Each directory contains `R.jsonl`, `S.jsonl`, `D.jsonl`, `P.jsonl`,
`batch_stats.csv`, and `run_manifest.json`. Compare the within-batch cosine
distribution before launching training. The CSV also carries mean question and
answer token lengths, initial answer NLL, and author composition for each
effective batch. If S, R, and D do not differ, stop and check the selected
representation rather than running the model grid.

The second-stage batch-order experiment is already supported. Keep a partition
method fixed and rebuild with `--batch-order similar` or
`--batch-order diverse`. Those files are named, for example,
`R_order-similar.jsonl`. Do this only after the first-stage grouping comparison.

## 4. Smoke and train

The launcher uses exactly two Accelerate processes with DeepSpeed ZeRO-3. It
does not use the older NPO shell script, whose direct `python src/train.py` call
does not launch the configured processes.

Run a no-save smoke with a complete manifest and stop after two optimizer
steps. The trainer then audits the consumed prefix while retaining the full
manifest contract:

```bash
scripts/representation_batching/run_npo_representation.sh \
  --manifest artifacts/representation_batching/seed-0/R.jsonl \
  --gpu 0,1 \
  --seed 0 \
  --max-steps 2 \
  --no-save
```

Then run the four paired arms for each seed:

```bash
for seed in 0 1 2; do
  for arm in R S D P; do
    scripts/representation_batching/run_npo_representation.sh \
      --manifest "artifacts/representation_batching/seed-${seed}/${arm}.jsonl" \
      --gpu 0,1 \
      --model-revision MODEL_COMMIT \
      --dataset-revision DATASET_COMMIT \
      --seed "${seed}"
  done
done
```

The default is full-parameter NPO plus retain NLL, beta 0.1, learning rate
2e-5, three epochs, constant schedule, bf16, and gradient checkpointing. The
launcher saves only the final model. Set `WANDB_MODE=online` if desired; offline
logging is the default.

Run the repository's TOFU evaluator separately on one GPU after training. The
current custom evaluator intentionally skips multi-process execution. Full,
Retain95, R/S/D/P, and any matched-forgetting checkpoints must use the same
evaluator code, prompts, and decoding configuration.

## 5. Required audit checks

Training aborts when the manifest disagrees with runtime world size, per-device
batch, accumulation, epoch count, forget row ids, or retain row count. The
ordered question-answer content is hashed as well, so a changed TOFU revision or
row order cannot silently reuse stale representations. The
custom dataloader is not passed through `accelerator.prepare`, because it has
already been split by global rank; Trainer still moves tensors to the correct
device. Each process compares the row ids emitted by the collator with its
planned microbatch and writes separate planned and observed JSONL logs. Rank 0
also archives the resolved Hydra config, package versions, git state, tracked
diff, and a copy of the experiment-specific source files.

At smoke time verify in both rank logs:

- ten local microsteps per optimizer step;
- no repeated or missing forget ids;
- planned and observed rank schedules agree exactly;
- equal local dataloader lengths;
- NPO forget loss near `2 log(2) / 0.1 = 13.8629` at initialization, before
  adding retain loss;
- NPO weight near 1 at initialization;
- the feature file hash and resolved Hydra config are archived with the run.

If the 8B active model plus frozen reference does not fit under this ZeRO-3
configuration, record the peak-memory failure and revise the resource setting.
Do not silently switch one arm to LoRA, CPU offload, another optimizer, or a
different effective batch.
