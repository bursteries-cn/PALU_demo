# PALU sparse hyperparameter analysis

This module analyzes `lr`, `alpha`, `top_k`, and `first_n` against:

- `model_utility` (larger is better)
- `forget_quality` (larger KS-test p-value is closer to the Retain distribution;
  it is not a forgetting percentage)
- `forget_Q_A_gibberish` (larger class-0/clean probability is better)
- `exact_memorization` (distance to the Retain baseline is minimized)

It is designed for incomplete, unbalanced grids. Raw metric values are never
imputed. Observed rankings, adjusted surrogate estimates, and proposed next runs
are kept separate in every output.

## Run

From the `PALU_demo` checkout:

```bash
python3 -m palu_hparam_analysis.cli \
  --root saves/unlearn \
  --config configs/analysis/palu_hparams.yaml \
  --out reports/hparam_analysis/forget05_llama31
```

The default is a fixed epoch-10 comparison. To inspect each run's last available
checkpoint instead:

```bash
python3 -m palu_hparam_analysis.cli --checkpoint-policy last
```

The report is a self-contained HTML file: `report.html`. It does not contact a
CDN or require a running web service. Static PNG/PDF/SVG figures and all
machine-readable tables are written beside it.

## Metadata and protocol rules

Metadata precedence is:

1. `<run>/.hydra/config.yaml`
2. `trainer_state.json` for checkpoint-step to epoch mapping
3. the run-directory suffix as a compatibility fallback

Conflicts are reported. By default only the largest identical protocol cohort is
modeled. The fingerprint includes model, split, trainer, attention implementation,
dtype, effective batch, target mode, gamma, retain-loss type, Retain reference,
gibberish-classifier settings, and generation settings. Use
`--allow-mixed-protocols` only when those differences are intentionally in scope.

## Statistical boundary

The primary surrogate is categorical Ridge regression with all four main effects
and all six pairwise interactions. Levels are not assumed to have a linear dose
response. Alpha is selected by grouped cross-validation and missing-cell intervals
come from bootstrap refits. Three- and four-way interactions are intentionally not
fit because sparse grids generally cannot identify them.

When scikit-learn is available, an ExtraTrees surrogate is used as a nonlinear
sensitivity check. Missing-cell run suggestions are emitted only if all four Ridge
models meet the configured grouped-CV R-squared threshold and every suggested cell
has observed support for all six parameter pairs.

`forget_quality` is modeled on a clipped logit scale solely as a response variable;
its scientific interpretation remains a KS-test p-value. Model validation describes
predictive adequacy, not causal identification.

## Main outputs

- `report.html`: filters, interactive observed scatter, audit summary, figures
- `tables/observed_configurations.csv`: replicate-aggregated observed results
- `tables/observed_pareto.csv`: observed four-metric Pareto set
- `tables/missing_combinations.csv`: planned but unavailable cells
- `tables/predicted_candidates.csv`: validation proposals, possibly empty by design
- `tables/adjusted_main_effects.csv`: estimated marginal means and intervals
- `tables/interaction_strengths.csv`: pairwise interaction RMS summaries
- `coverage_audit.json`, `model_diagnostics.json`, `provenance.json`

## Practical interpretation

The first row of the observed recommendation is the Pareto configuration with the
strongest replicate support and smallest mean across-seed metric range, followed
by its worst and mean metric percentiles. When no configuration has replicated
seeds, the scale-free percentile tie-breaker is used directly. These rules remain
secondary to constraints and Pareto membership. Set
scientifically meaningful thresholds in `configs/analysis/palu_hparams.yaml` before
using the result as an experiment-selection rule.
