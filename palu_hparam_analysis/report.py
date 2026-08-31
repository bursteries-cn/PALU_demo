from __future__ import annotations

import csv
import html
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .coverage import coverage_audit
from .effects import fit_all_metrics, predicted_candidates
from .ingest import (
    PROTOCOL_FIELDS,
    collect_records,
    filter_records,
    select_checkpoints,
    select_protocol_cohort,
)
from .pareto import (
    aggregate_configurations,
    load_baselines,
    rank_observed,
    resolve_em_target,
)
from .plots import (
    plot_coverage,
    plot_epoch_trajectories,
    plot_interactions,
    plot_main_effects,
    plot_model_diagnostics,
    plot_parallel_coordinates,
    plot_tradeoffs,
)
from .schema import FACTORS, METRICS, METRIC_LABELS, __version__


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_serializable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(_serializable(value), ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], preferred: Iterable[str] = ()) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {key for row in rows for key in row}
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fields})


def _figure_png(paths: list[str], report_dir: Path) -> Optional[str]:
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() == ".png":
            return path.relative_to(report_dir).as_posix()
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() == ".svg":
            return path.relative_to(report_dir).as_posix()
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{number:.{digits}g}"


def _warning_list(warnings: list[str]) -> str:
    if not warnings:
        return '<p class="ok">No ingestion or protocol warnings.</p>'
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warnings) + "</ul>"


def _diagnostic_table(models: dict[str, dict[str, Any]]) -> str:
    rows = []
    for metric in METRICS:
        diagnostics = models.get(metric, {}).get("diagnostics", {})
        selected = diagnostics.get("cv", {}).get("selected", {})
        tree = diagnostics.get("tree_sensitivity", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(METRIC_LABELS[metric])}</td>"
            f"<td><span class='status {html.escape(str(diagnostics.get('status', 'not_fitted')))}'>"
            f"{html.escape(str(diagnostics.get('status', 'not fitted')))}</span></td>"
            f"<td>{diagnostics.get('n_rows', 0)}</td>"
            f"<td>{_fmt(selected.get('rmse'))}</td>"
            f"<td>{_fmt(selected.get('r2'))}</td>"
            f"<td>{html.escape(str(tree.get('status', '—')))}</td>"
            f"<td>{_fmt(tree.get('ridge_grid_correlation'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _baseline_table(baselines: dict[str, Any]) -> str:
    rows = []
    for name in ("full", "retain"):
        values = baselines.get(name, {})
        rows.append(
            "<tr>"
            f"<td>{name.title()}</td>"
            + "".join(f"<td>{_fmt(values.get(metric), 6)}</td>" for metric in METRICS)
            + "</tr>"
        )
    return "".join(rows)


def _best_observed(rows: list[dict[str, Any]]) -> str:
    candidates = [
        row
        for row in rows
        if row.get("passes_constraints") and row.get("pareto_observed")
    ]
    if not candidates:
        return (
            "<p>No complete, constraint-passing observed configuration is available. "
            "This report therefore does not name an observed best configuration.</p>"
        )
    row = candidates[0]
    factor_text = ", ".join(f"{factor}={html.escape(str(row[factor]))}" for factor in FACTORS)
    metric_text = ", ".join(
        f"{METRIC_LABELS[metric]}={_fmt(row.get(metric), 5)}" for metric in METRICS
    )
    return (
        f"<p><strong>Balanced observed recommendation:</strong> {factor_text}</p>"
        f"<p>{metric_text}</p>"
        "<p class='note'>This is an observed Pareto configuration selected by maximin percentile; "
        "it is not a causal mechanism claim.</p>"
    )


def build_html_report(
    output_path: Path,
    observed: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    audit: dict[str, Any],
    baselines: dict[str, Any],
    em_target: Optional[float],
    models: dict[str, dict[str, Any]],
    figures: list[dict[str, str]],
    warnings: list[str],
    config: dict[str, Any],
) -> None:
    observed_json = json.dumps(_serializable(observed), ensure_ascii=False).replace("</", "<\\/")
    predicted_json = json.dumps(_serializable(predicted), ensure_ascii=False).replace("</", "<\\/")
    figure_html = "".join(
        f"<figure><img src='{html.escape(item['path'])}' alt='{html.escape(item['title'])}'>"
        f"<figcaption>{html.escape(item['title'])}</figcaption></figure>"
        for item in figures
    )
    filters = config.get("filters", {})
    context = ", ".join(
        f"{key}={value}" for key, value in filters.items() if value is not None
    )
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PALU hyperparameter analysis</title>
<style>
:root {{ --blue:#0072B2; --orange:#E69F00; --green:#009E73; --pink:#CC79A7; --ink:#17212b; --muted:#607080; --line:#d8e0e8; --paper:#ffffff; --wash:#f4f7fa; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--wash); font:14px/1.5 Arial,Helvetica,sans-serif; }}
header {{ background:linear-gradient(120deg,#073763,#0072B2); color:white; padding:32px max(24px,calc((100vw - 1180px)/2)); }}
header h1 {{ margin:0 0 8px; font-size:28px; }} header p {{ margin:4px 0; opacity:.92; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
section {{ background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:22px; margin-bottom:18px; box-shadow:0 3px 12px rgba(30,60,90,.05); }}
h2 {{ margin:0 0 14px; font-size:20px; }} h3 {{ margin-top:20px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:12px; }}
.card {{ padding:14px; border-radius:9px; background:#f7fafc; border-left:4px solid var(--blue); }}
.card b {{ display:block; font-size:22px; }} .card span {{ color:var(--muted); font-size:12px; }}
.ok {{ color:var(--green); }} .note {{ color:var(--muted); font-size:12px; }}
.status {{ padding:2px 7px; border-radius:10px; background:#edf1f5; }} .status.validated {{ background:#dff3e8; color:#006b45; }} .status.exploratory {{ background:#fff0cc; color:#805700; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }}
figure {{ margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:white; }}
figure img {{ width:100%; display:block; }} figcaption {{ padding:9px 12px; color:var(--muted); }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; margin:12px 0; align-items:end; }}
label {{ font-size:12px; color:var(--muted); }} select {{ display:block; min-width:120px; padding:7px; border:1px solid var(--line); border-radius:5px; background:white; }}
#plotWrap {{ position:relative; overflow-x:auto; }} #scatter {{ border:1px solid var(--line); background:white; max-width:100%; }}
#tooltip {{ position:absolute; display:none; pointer-events:none; background:rgba(23,33,43,.94); color:white; padding:8px 10px; border-radius:5px; font-size:12px; max-width:330px; white-space:pre-line; }}
.tableWrap {{ overflow:auto; max-height:560px; border:1px solid var(--line); }} table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th {{ position:sticky; top:0; background:#eaf0f6; text-align:left; z-index:1; }} th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; white-space:nowrap; }}
tr.pareto td:first-child {{ border-left:4px solid var(--orange); }} tr:hover {{ background:#f2f7fb; }}
code {{ background:#eef2f6; padding:2px 5px; border-radius:4px; }}
@media(max-width:700px) {{ .gallery {{ grid-template-columns:1fr; }} main {{ padding:12px; }} }}
</style>
</head>
<body>
<header>
  <h1>PALU sparse four-factor analysis</h1>
  <p>{html.escape(context or 'No filters configured')}</p>
  <p>Generated {html.escape(generated)} · module v{__version__}</p>
</header>
<main>
<section>
  <h2>Data and decision summary</h2>
  <div class="cards">
    <div class="card"><b>{audit['observed_configurations']}</b><span>observed configurations</span></div>
    <div class="card"><b>{audit['planned_combinations']}</b><span>planned grid combinations</span></div>
    <div class="card"><b>{audit['coverage_fraction']:.1%}</b><span>design coverage</span></div>
    <div class="card"><b>{audit['complete_four_metric_rows']}</b><span>complete four-metric rows</span></div>
    <div class="card"><b>{len([row for row in observed if row.get('pareto_observed')])}</b><span>observed Pareto configurations</span></div>
    <div class="card"><b>{len(predicted)}</b><span>predicted next-run candidates</span></div>
  </div>
  <h3>Observed recommendation</h3>
  {_best_observed(observed)}
  <p><strong>Exact-memorization target:</strong> {_fmt(em_target, 6)} (Retain baseline when available).</p>
  <h3>Reference baselines</h3>
  <div class="tableWrap"><table><thead><tr><th>Reference</th>{''.join(f'<th>{html.escape(METRIC_LABELS[metric])}</th>' for metric in METRICS)}</tr></thead><tbody>{_baseline_table(baselines)}</tbody></table></div>
</section>

<section>
  <h2>Interactive observed-result explorer</h2>
  <p class="note">Orange rings are observed four-dimensional Pareto configurations. Predictions never appear in this observed plot.</p>
  <div class="controls">
    <label>X metric<select id="xMetric"></select></label>
    <label>Y metric<select id="yMetric"></select></label>
    <label>Learning rate<select id="f_lr"></select></label>
    <label>Alpha<select id="f_alpha"></select></label>
    <label>Top-K<select id="f_top_k"></select></label>
    <label>Initial-N<select id="f_first_n"></select></label>
  </div>
  <div id="plotWrap"><canvas id="scatter" width="1040" height="520"></canvas><div id="tooltip"></div></div>
  <h3>Observed configurations</h3>
  <div class="tableWrap"><table id="resultTable"></table></div>
</section>

<section>
  <h2>Surrogate-model validation</h2>
  <p>Pairwise surfaces are exploratory unless the grouped cross-validation status is <code>validated</code>. FQ remains a KS-test p-value, not a forgetting percentage.</p>
  <div class="tableWrap"><table><thead><tr><th>Metric</th><th>Status</th><th>n</th><th>CV RMSE</th><th>CV R²</th><th>Tree check</th><th>Ridge/tree grid r</th></tr></thead><tbody>{_diagnostic_table(models)}</tbody></table></div>
</section>

<section>
  <h2>Predicted next-run candidates</h2>
  <p class="note">This table is populated only when all four response models pass grouped cross-validation and each missing cell has observed support for every parameter pair. Values are proposals to validate, not observed best results.</p>
  <div class="tableWrap"><table id="predictedTable"></table></div>
</section>

<section><h2>Static scientific figures</h2><div class="gallery">{figure_html}</div></section>

<section>
  <h2>Audit warnings and interpretation boundary</h2>
  {_warning_list(warnings)}
  <p class="note">Observed Pareto results are descriptive under the selected protocol. Adjusted effects and missing-cell predictions are model-based estimates. Predicted candidates are proposals for controlled reruns, never validated best configurations.</p>
</section>

<section>
  <h2>Machine-readable outputs</h2>
  <p>See <code>tables/</code> for observed runs, aggregated configurations, Pareto rows, missing combinations, protocol cohorts and predicted candidates; see <code>model_diagnostics.json</code> and <code>provenance.json</code> for model and audit details.</p>
</section>
</main>
<script>
const observed = {observed_json};
const predicted = {predicted_json};
const metrics = {json.dumps(list(METRICS))};
const labels = {json.dumps(METRIC_LABELS)};
const factors = {json.dumps(list(FACTORS))};
const canvas = document.getElementById('scatter');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
let plotted = [];
function unique(field) {{ return [...new Set(observed.map(r => String(r[field])))].sort((a,b)=>Number(a)-Number(b)); }}
function fillSelect(id, values, includeAll=true) {{ const el=document.getElementById(id); el.innerHTML=''; if(includeAll) el.add(new Option('All','*')); values.forEach(v=>el.add(new Option(v,v))); }}
fillSelect('xMetric',metrics,false); fillSelect('yMetric',metrics,false); document.getElementById('xMetric').value='model_utility'; document.getElementById('yMetric').value='forget_quality';
factors.forEach(f=>fillSelect('f_'+f,unique(f),true));
function filtered() {{ return observed.filter(r => factors.every(f => {{ const v=document.getElementById('f_'+f).value; return v==='*'||String(r[f])===v; }})); }}
function transformed(metric,value) {{ if(metric==='forget_quality') return Math.log10(Math.max(Number(value),1e-16)); return Number(value); }}
function draw() {{
  const xMetric=document.getElementById('xMetric').value, yMetric=document.getElementById('yMetric').value;
  const rows=filtered().filter(r=>Number.isFinite(Number(r[xMetric]))&&Number.isFinite(Number(r[yMetric])));
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#fff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const margin={{left:82,right:25,top:25,bottom:70}}, width=canvas.width-margin.left-margin.right, height=canvas.height-margin.top-margin.bottom;
  const xs=rows.map(r=>transformed(xMetric,r[xMetric])), ys=rows.map(r=>transformed(yMetric,r[yMetric]));
  const extent=(vals)=>{{ if(!vals.length)return [0,1]; let lo=Math.min(...vals),hi=Math.max(...vals); if(lo===hi){{lo-=.5;hi+=.5;}} const p=(hi-lo)*.08; return [lo-p,hi+p]; }};
  const [x0,x1]=extent(xs),[y0,y1]=extent(ys); const px=v=>margin.left+(v-x0)/(x1-x0)*width, py=v=>margin.top+height-(v-y0)/(y1-y0)*height;
  ctx.strokeStyle='#8a98a6'; ctx.lineWidth=1; ctx.beginPath();ctx.moveTo(margin.left,margin.top);ctx.lineTo(margin.left,margin.top+height);ctx.lineTo(margin.left+width,margin.top+height);ctx.stroke();
  ctx.fillStyle='#17212b';ctx.font='13px Arial';ctx.textAlign='center';ctx.fillText(labels[xMetric]+(xMetric==='forget_quality'?' (log10)':''),margin.left+width/2,canvas.height-18);
  ctx.save();ctx.translate(20,margin.top+height/2);ctx.rotate(-Math.PI/2);ctx.fillText(labels[yMetric]+(yMetric==='forget_quality'?' (log10)':''),0,0);ctx.restore();
  plotted=[]; rows.forEach((r,i)=>{{ const x=px(xs[i]),y=py(ys[i]);ctx.beginPath();ctx.arc(x,y,r.pareto_observed?7:4,0,Math.PI*2);ctx.fillStyle=r.pareto_observed?'#fff':'#0072B2';ctx.fill();ctx.strokeStyle=r.pareto_observed?'#E69F00':'#0072B2';ctx.lineWidth=r.pareto_observed?2.4:1;ctx.stroke();plotted.push({{x,y,r}}); }});
  renderTable(rows);
}}
function renderTable(rows) {{
  const cols=['pareto_observed',...factors,...metrics,'n_runs','replicate_supported','mean_metric_replicate_range','balanced_min_percentile']; const table=document.getElementById('resultTable');
  table.innerHTML='<thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr class="'+(r.pareto_observed?'pareto':'')+'">'+cols.map(c=>'<td>'+format(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody>';
}}
function renderPredicted() {{
  const table=document.getElementById('predictedTable');
  if(!predicted.length) {{ table.innerHTML='<tbody><tr><td>No validated missing-cell recommendation was emitted. Inspect model diagnostics and coverage.</td></tr></tbody>'; return; }}
  const cols=['pareto_predicted',...factors,...metrics,'minimum_pair_support','mean_prediction_interval_width','balanced_min_percentile'];
  table.innerHTML='<thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+predicted.map(r=>'<tr class="'+(r.pareto_predicted?'pareto':'')+'">'+cols.map(c=>'<td>'+format(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody>';
}}
function format(v) {{ if(v===null||v===undefined)return '—'; if(typeof v==='number')return Number(v).toPrecision(5); if(v===true)return '●'; if(v===false)return ''; return String(v); }}
canvas.addEventListener('mousemove',e=>{{ const rect=canvas.getBoundingClientRect(),sx=canvas.width/rect.width,sy=canvas.height/rect.height,x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy; let best=null,dist=Infinity;plotted.forEach(p=>{{const d=(p.x-x)**2+(p.y-y)**2;if(d<dist){{dist=d;best=p;}}}});if(best&&dist<140){{const r=best.r;tooltip.style.display='block';tooltip.style.left=(e.clientX-rect.left+14)+'px';tooltip.style.top=(e.clientY-rect.top+14)+'px';tooltip.textContent=factors.map(f=>f+'='+r[f]).join(', ')+'\n'+metrics.map(m=>labels[m]+'='+format(r[m])).join('\n');}}else tooltip.style.display='none';}});
document.querySelectorAll('select').forEach(el=>el.addEventListener('change',draw)); renderPredicted(); draw();
</script>
</body></html>"""
    output_path.write_text(html_text, encoding="utf-8")


def _effect_tables(models: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows = []
    interaction_rows = []
    for metric, result in models.items():
        for factor, entries in result.get("main_effects", {}).items():
            for entry in entries:
                main_rows.append({"metric": metric, "factor": factor, **entry})
        for pair in result.get("pair_effects", []):
            interaction_rows.append(
                {
                    "metric": metric,
                    "left": pair["left"],
                    "right": pair["right"],
                    "interaction_rms": pair["interaction_rms"],
                }
            )
    return main_rows, interaction_rows


def run_analysis(
    result_root: Path,
    config: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_records, warnings = collect_records(result_root)
    filtered = filter_records(all_records, config)
    cohort_records, protocol_excluded, protocol_cohorts = select_protocol_cohort(filtered, config)
    if protocol_excluded:
        warnings.append(
            f"excluded {len(protocol_excluded)} rows outside the largest protocol cohort; see tables/protocol_cohorts.csv"
        )
    selected = select_checkpoints(cohort_records, config)
    if filtered and not selected:
        checkpoint = config.get("checkpoint", {})
        warnings.append(
            f"no rows matched checkpoint policy {checkpoint}; available epochs: "
            + ", ".join(str(value) for value in sorted({row.get('epoch') for row in cohort_records}))
        )
    observed = aggregate_configurations(selected)
    baselines, baseline_warnings = load_baselines(str(repo_root), config)
    warnings.extend(baseline_warnings)
    em_target = resolve_em_target(config, baselines)
    if em_target is None:
        warnings.append(
            "Retain exact_memorization baseline is unavailable; EM falls back to lower-is-better for ranking"
        )
    observed = rank_observed(observed, config, em_target)
    audit = coverage_audit(observed, config)
    models = fit_all_metrics(observed, config)
    for metric, result in models.items():
        tree = result.get("diagnostics", {}).get("tree_sensitivity", {})
        correlation = tree.get("ridge_grid_correlation")
        if tree.get("status") == "ok" and correlation is not None and float(correlation) < 0.7:
            warnings.append(
                f"{metric}: Ridge and ExtraTrees missing-cell surfaces disagree "
                f"(grid correlation={float(correlation):.3f}); treat adjusted effects as unstable"
            )
    predicted, prediction_warnings = predicted_candidates(observed, models, config, em_target)
    warnings.extend(prediction_warnings)

    preferred = [*FACTORS, "epoch", *METRICS, "n_runs", "pareto_observed"]
    write_csv(tables_dir / "all_filtered_checkpoints.csv", cohort_records, preferred)
    write_csv(tables_dir / "selected_checkpoint_rows.csv", selected, preferred)
    write_csv(tables_dir / "observed_configurations.csv", observed, preferred)
    write_csv(
        tables_dir / "observed_pareto.csv",
        [row for row in observed if row.get("pareto_observed")],
        preferred,
    )
    write_csv(tables_dir / "missing_combinations.csv", audit["missing_combinations"], FACTORS)
    write_csv(tables_dir / "outside_search_space.csv", audit["outside_search_space"], preferred)
    write_csv(tables_dir / "protocol_cohorts.csv", protocol_cohorts, [*PROTOCOL_FIELDS, "n_rows"])
    write_csv(tables_dir / "excluded_protocol_rows.csv", protocol_excluded, preferred)
    write_csv(tables_dir / "predicted_candidates.csv", predicted, preferred)
    main_effects, interactions = _effect_tables(models)
    write_csv(tables_dir / "adjusted_main_effects.csv", main_effects)
    write_csv(tables_dir / "interaction_strengths.csv", interactions)

    write_json(output_dir / "model_diagnostics.json", {
        metric: result.get("diagnostics", {}) for metric, result in models.items()
    })
    write_json(output_dir / "coverage_audit.json", audit)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module_version": __version__,
        "python": platform.python_version(),
        "result_root": str(result_root.resolve()),
        "repo_root": str(repo_root.resolve()),
        "config": config,
        "baselines": baselines,
        "exact_memorization_target": em_target,
        "counts": {
            "all_records": len(all_records),
            "filtered_records": len(filtered),
            "protocol_cohort_records": len(cohort_records),
            "selected_checkpoint_records": len(selected),
            "observed_configurations": len(observed),
            "predicted_candidates": len(predicted),
        },
        "warnings": warnings,
    }
    write_json(output_dir / "provenance.json", provenance)

    figure_entries = []
    coverage_paths = plot_coverage(audit, figures_dir / "coverage", config)
    coverage_image = _figure_png(coverage_paths, output_dir)
    if coverage_image:
        figure_entries.append({"title": "Pairwise design coverage", "path": coverage_image})
    tradeoff_paths = plot_tradeoffs(observed, em_target, figures_dir / "observed_tradeoffs", config)
    tradeoff_image = _figure_png(tradeoff_paths, output_dir)
    if tradeoff_image:
        figure_entries.append({"title": "Observed four-metric trade-offs", "path": tradeoff_image})
    parallel_paths = plot_parallel_coordinates(
        observed, em_target, figures_dir / "observed_parallel_coordinates", config
    )
    parallel_image = _figure_png(parallel_paths, output_dir)
    if parallel_image:
        figure_entries.append(
            {"title": "Observed scale-free parallel coordinates", "path": parallel_image}
        )
    if cohort_records:
        epoch_paths = plot_epoch_trajectories(cohort_records, figures_dir / "epoch_trajectories", config)
        epoch_image = _figure_png(epoch_paths, output_dir)
        if epoch_image:
            figure_entries.append({"title": "Checkpoint trajectories", "path": epoch_image})
    diagnostics_paths = plot_model_diagnostics(models, figures_dir / "model_diagnostics", config)
    diagnostics_image = _figure_png(diagnostics_paths, output_dir)
    if diagnostics_image:
        figure_entries.append({"title": "Surrogate-model diagnostics", "path": diagnostics_image})
    for metric, result in models.items():
        if "main_effects" not in result:
            continue
        main_paths = plot_main_effects(result, figures_dir / f"main_effects_{metric}", config)
        main_image = _figure_png(main_paths, output_dir)
        if main_image:
            figure_entries.append({"title": f"Adjusted main effects: {METRIC_LABELS[metric]}", "path": main_image})
        pair_paths = plot_interactions(result, figures_dir / f"interactions_{metric}", config)
        pair_image = _figure_png(pair_paths, output_dir)
        if pair_image:
            figure_entries.append({"title": f"Pairwise interactions: {METRIC_LABELS[metric]}", "path": pair_image})

    build_html_report(
        output_dir / "report.html",
        observed,
        predicted,
        audit,
        baselines,
        em_target,
        models,
        figure_entries,
        warnings,
        config,
    )
    manifest = {
        "report": str((output_dir / "report.html").resolve()),
        "provenance": str((output_dir / "provenance.json").resolve()),
        "tables": str(tables_dir.resolve()),
        "figures": str(figures_dir.resolve()),
        "counts": provenance["counts"],
        "warnings": warnings,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
