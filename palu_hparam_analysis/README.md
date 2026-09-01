# PALU sparse hyperparameter record and guidance

This module keeps a compact record of PALU experiments over `lr`, `alpha`,
`top_k`, and `first_n`, then uses only the available results to indicate which
values currently look more promising.

It is intentionally designed for an incomplete grid. It does **not** require a
full Cartesian sweep, fill missing metric values, or label an untested
configuration as the best result.

## Run

From the `PALU_demo` checkout:

```bash
python3 -m palu_hparam_analysis.cli \
  --root saves/unlearn \
  --config configs/analysis/palu_hparams.yaml \
  --out reports/hparam_analysis/forget05_llama31
```

The default compares epoch 10. To record the last evaluated checkpoint from each
run instead:

```bash
python3 -m palu_hparam_analysis.cli --checkpoint-policy last
```

Models are analyzed separately. For example, use `--model
Llama-2-7b-chat-hf` together with Llama-2 Full/Retain baseline paths in a
model-specific analysis config.

## What the report answers

1. Which parameter combinations have a result at the selected checkpoint?
2. Which combinations have runs, but not the selected checkpoint?
3. Which planned combinations have not been run?
4. Among tested levels, which one or two values currently have the best median
   relative performance?
5. Which small set of one-parameter neighboring experiments would most directly
   refine the current promising region?

The report also includes a 4-by-4 raw-value figure: columns are the four
hyperparameters and rows are the four metrics. Blue points are individual
observed configurations; the orange line connects per-level medians across
adjacent observed levels, with interquartile-range error bars. Missing levels are
not imputed, and a completely unavailable metric is labeled explicitly.

The compact guidance score is the equal-weight mean of direction-aligned metric
percentiles within the current observed set. A metric participates when it has at
least two observations and is available for at least half of the configurations.
If a participating metric is missing from one configuration, that configuration
receives zero credit for the missing value. The score is a search aid, not a
paper metric or causal effect.

`forget_quality` remains a KS-test p-value, not a forgetting percentage.
`exact_memorization` is scored by distance to the Retain baseline when that
baseline is available; otherwise lower is treated as better and the report emits
a warning.

## Main outputs

- `report.html`: compact guidance, value summaries, next local probes, and a
  filterable experiment ledger
- `figures/metric_by_parameter_grid.*`: 16 raw metric-by-parameter scatter and
  median-line panels
- `tables/experiment_ledger.csv`: one row per planned combination with status
  `tested`, `other_checkpoint`, or `missing`
- `tables/observed_configurations.csv`: observed metrics aggregated by parameter
  combination
- `tables/parameter_value_summary.csv`: support counts and metric medians for
  every configured value
- `tables/recommended_next_experiments.csv`: a short list of untested one-step
  neighbors of the strongest observed configurations
- `coverage_audit.json` and `provenance.json`: machine-readable audit context

## Metadata and comparison boundary

Metadata precedence is:

1. `<run>/.hydra/config.yaml`
2. `trainer_state.json` for checkpoint-step to epoch mapping
3. the run-directory suffix as a compatibility fallback

By default the report uses only the largest identical protocol cohort. Its
fingerprint includes model, split, trainer, attention implementation, dtype,
effective batch, loss settings, classifier settings, and generation settings.
This prevents results from different models or evaluation protocols from being
silently combined.
