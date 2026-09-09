"""Compact seed-by-arm table and four-panel metric figure, with source provenance."""
from __future__ import annotations
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from representation_batching.report import METRICS, build_report, finite, fmt, read_json, write_csv, report_lock

ARMS = ("R", "S", "D", "P")
LABELS = {"exact_memorization": "Exact memorization",
          "forget_Q_A_gibberish": "Fluency (clean probability)",
          "forget_quality": "Forget quality (KS p-value)", "model_utility": "Model utility"}
COLORS = {"R":"#4277a5", "S":"#de842b", "D":"#228978", "P":"#a266a7"}
MARKERS = {"R":"o", "S":"s", "D":"^", "P":"D"}


def manual_rows(path):
    rows=[]
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            if record["arm"] not in ARMS:
                raise ValueError(f"Unsupported arm: {record['arm']}")
            row = {"seed":int(record["seed"]), "arm":record["arm"]+"/random",
                   "run":record.get("run") or "user_reported", "run_dir":"",
                   "summary_path":str(Path(path).resolve()), "training_status":"user_reported",
                   "evaluation_status":"user_reported", "protocol":None, "evaluation_protocol":None,
                   "eligible":False}
            for metric in METRICS:
                value=record.get(metric)
                if value not in (None, ""):
                    value=float(value)
                    if not finite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid {metric}: {value}")
                    row[metric]=value
            rows.append(row)
    return rows


def select_cells(rows, seeds=None):
    groups=defaultdict(list)
    for row in rows:
        if row.get("training_status") not in ("completed", "user_reported"):
            continue
        if row.get("arm") not in [a+"/random" for a in ARMS]:
            continue
        groups[(int(row["seed"]), row["arm"].split("/")[0])].append(row)
    selected, warnings={}, []
    seed_values=sorted(set(int(r["seed"]) for r in rows) | set(seeds or []))
    for seed in seed_values:
        for arm in ARMS:
            matches=groups.get((seed,arm),[])
            if len(matches)==1:
                selected[(seed,arm)]=matches[0]
            elif len(matches)>1:
                warnings.append(f"seed={seed}, {arm}: {len(matches)} 个完整运行/手填记录；该单元格留空，请用 --root 指定所需运行或移除重复手填记录。")
    wide=[]
    for seed in seed_values:
        for metric in METRICS:
            record={"seed":seed,"metric":metric}
            for arm in ARMS:
                row=selected.get((seed,arm),{})
                record[arm]=row.get(metric) if finite(row.get(metric)) else None
                record[arm+"_source"]=row.get("summary_path") or ""
                record[arm+"_protocol"]=str(row.get("protocol") or "unknown")+" / "+str(row.get("evaluation_protocol") or "unknown")
                record[arm+"_status"]=row.get("evaluation_status", "missing_or_ambiguous")
            wide.append(record)
    return wide, selected, warnings


def draw(selected, seeds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    fig, axes=plt.subplots(2,2,figsize=(11,7.6))
    fig.subplots_adjust(top=.82,bottom=.16,wspace=.2,hspace=.5)
    display_seeds=seeds or [0]
    for ax, metric in zip(axes.flat, METRICS):
        for arm in ARMS:
            cohorts=defaultdict(list)
            for (seed, selected_arm), row in selected.items():
                if selected_arm==arm and finite(row.get(metric)):
                    cohorts[(row.get("protocol"),row.get("evaluation_protocol"), bool(row.get("eligible")))].append((seed,row[metric]))
            for (training, evaluation, verified), values in cohorts.items():
                series={s:v for s,v in values}
                y=[series.get(seed,np.nan) for seed in display_seeds]
                # Unknown protocols are displayed as points only. A changed protocol
                # splits the line, and NaNs break it at missing/ambiguous cells.
                offset=(ARMS.index(arm)-1.5)*.075
                ax.plot(np.arange(len(display_seeds))+offset, y, color=COLORS[arm], marker=MARKERS[arm],
                    markersize=6, linewidth=1.6, linestyle="-" if verified and training and evaluation else "None",
                    markerfacecolor=COLORS[arm] if verified else "white", markeredgewidth=1.6)
        ax.set_title(LABELS[metric],fontsize=11,loc="left",pad=10)
        ax.set_xticks(range(len(display_seeds)),[str(s) for s in display_seeds])
        ax.set_xlabel("Seed (independent repeats)")
        ax.set_ylim(-.035,1.05)
        ax.set_xlim(-.35,max(len(display_seeds)-1,.1)+.35)
        ax.grid(axis="y",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
        if not any(finite(row.get(metric)) for row in selected.values()):
            ax.text(.5,.5,"No evaluated runs",transform=ax.transAxes,ha="center",color="#777")
    handles=[Line2D([],[],color=COLORS[a],marker=MARKERS[a],label=a) for a in ARMS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
    fig.suptitle("NPO grouping experiments by seed",fontsize=16,y=.985)
    fig.text(.5,.035,"Open markers: reported/unverified. Lines: same verified protocol only.\nSmall horizontal offsets separate arms. Missing values break lines.\nKS p-values are not forgetting percentages.",fontsize=9,ha="center")
    for suffix in ("png","pdf","svg"):
        fig.savefig(out / f"seed_comparison.{suffix}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def export_comparison(rows, out, seeds=None, plots=True):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    wide, selected, warnings=select_cells(rows,seeds)
    write_csv(out / "seed_comparison.csv",wide)
    write_csv(out / "seed_runs.csv",rows)
    plot_created=False
    if plots:
        try:
            draw(selected,sorted({r["seed"] for r in wide}),out)
            plot_created=True
        except ImportError as exc:
            warnings.append(f"CSV/HTML 已生成；缺少绘图库，未生成折线图：{exc}。安装项目 requirements.txt 中的 matplotlib。")
    def value_cell(record,arm):
        value=html.escape(fmt(record[arm]))
        status=html.escape(record[arm+"_status"])
        source=html.escape(record[arm+"_source"],quote=True)
        return f'<td title="{source}">{value}<small>{status}</small></td>'
    rows_html=''.join('<tr><td>'+str(r['seed'])+'</td><td>'+html.escape(r['metric'])+'</td>'+''.join(value_cell(r,a) for a in ARMS)+'</tr>' for r in wide)
    image='<img src="seed_comparison.png" alt="四种分组的跨 seed 指标图">' if plot_created else ''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>各 seed 四组结果</title><style>body{font:15px/1.6 system-ui;margin:32px;color:#243747}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dde4e9;text-align:left}th{background:#eef4f7}small{display:block;font-size:10px;color:#777}img{width:100%;max-width:1200px}a{color:#267699}.scroll{overflow:auto}</style><h1>各 seed 的 R / S / D / P</h1><p>一行对应一个 seed 的一个指标。— 表示缺失或重复待选择；手填/未核验结果明确标记，不自动视为严格对照。</p><p>EM 应结合匹配 Retain 参考解释；FQ 是 KS p 值，不是遗忘百分比。折线仅连接同一已核验协议下的运行；seed 编号没有大小优劣含义。</p><p><a href="seed_comparison.csv">结果表 CSV</a> · <a href="seed_runs.csv">来源与配置 CSV</a></p>'''+image+'<div class="scroll"><table><thead><tr><th>seed</th><th>metric</th>'+''.join('<th>'+a+'</th>' for a in ARMS)+'</tr></thead><tbody>'+rows_html+'</tbody></table></div><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in warnings)+'</ul></html>'
    (out / "seed_comparison.html").write_text(page,encoding="utf-8")
    return wide,warnings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",action="append",type=Path,help="Repeat for multiple training roots")
    parser.add_argument("--results",type=Path,help="Use an existing report results.json instead of scanning")
    parser.add_argument("--metrics-csv",type=Path,help="Explicitly labelled user-reported metrics (seed, arm, four metrics)")
    parser.add_argument("--out",type=Path,default=Path("reports/representation_batching"))
    parser.add_argument("--seeds",nargs="+",type=int)
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    if args.results and args.root:
        parser.error("Use --results or --root, not both")
    with report_lock(args.out):
        if args.results:
            rows=read_json(args.results)["runs"]
        elif args.root or not args.metrics_csv:
            roots=args.root or [Path("saves/unlearn")]
            payload=build_report({"roots":[str(p.resolve()) for p in roots],"expected_seeds":args.seeds or [0,1,2]},args.out.resolve())
            rows=payload["runs"]
            for warning in payload["warnings"]:
                print("WARNING:",warning)
        else:
            rows=[]
        if args.metrics_csv:
            rows+=manual_rows(args.metrics_csv)
        wide,warnings=export_comparison(rows,args.out,args.seeds,not args.no_plot)
    print("seed\tmetric\tR\tS\tD\tP")
    for row in wide:
        print('\t'.join(fmt(row[k]) for k in ("seed","metric",*ARMS)))
    for warning in warnings:
        print("WARNING:",warning)
    print(f"Report: {args.out.resolve() / 'seed_comparison.html'}")


if __name__=="__main__":
    main()
