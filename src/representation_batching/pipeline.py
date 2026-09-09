"""Run one paired seed through manifest generation, R/S/D/P training and evaluation."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone

from representation_batching.report import METRICS, build_report, collect_run, read_json, write_csv
from representation_batching.seed_comparison import export_comparison
from representation_batching.local_dataset import resolve_local_dataset_config

ROOT=Path(__file__).resolve().parents[2]
ARMS=("R","S","D","P")
# The first released pipeline included reporting/orchestration code in its hash.
# These files do not change the training/evaluation numerical protocol.
CONTROL_FILES={"src/representation_batching/pipeline.py",
               "src/representation_batching/report.py",
               "src/representation_batching/seed_comparison.py"}
LEGACY_PIPELINE_COMMITS=("a5d9198698d99f1ec0699e99bb3e17b0cb039dcf",
                         "b1d2b1ebdb712ebc25b794091e9f726a75a54985")
# Exact, reviewed metadata-only eval.py change: snapshot the request before the
# loader consumes its path/dtype. Do not exempt future evaluator code changes.
EVAL_PROVENANCE_FIX_SHA256="9e8979afd97614f2cd86060f2b0c365c9ad2f0d2a3bb1a1d41751254536ec602"
LEGACY_EVAL_SHA256="b23d348348ce9b1f2b34c441e8570f020873ad180972e1e059e119297d2438b2"


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def path_from_root(value,root):
    path=Path(value).expanduser()
    return (root/path).resolve() if not path.is_absolute() else path.resolve()


def load_settings(config_path,root=ROOT):
    raw=read_json(config_path)
    settings=dict(raw)
    feature=raw.get("features")
    if not feature:
        conventional=root / "artifacts/representation_batching/features-forget05/features.npz"
        candidates=[conventional] if conventional.exists() else sorted((root / "artifacts/representation_batching").rglob("features.npz"))
        if len(candidates)!=1:
            raise ValueError(f"找到 {len(candidates)} 份 features.npz。请在 {config_path} 的 features 字段填写原特征文件路径。")
        feature=str(candidates[0])
    settings["features"]=str(path_from_root(feature,root))
    feature_meta=read_json(Path(settings["features"]).parent / "feature_manifest.json")
    actual_hash=hash_file(settings["features"])
    if actual_hash!=feature_meta.get("feature_sha256"):
        raise ValueError("features.npz 与 feature_manifest.json 的哈希不一致")
    if feature_meta.get("dataset_config")!="forget05":
        raise ValueError("本脚本对应当前 forget05/retain95 实验；其他 split 需要独立配置和训练入口")
    settings["feature_key"]=raw.get("feature_key","block_22_question_last")
    if settings["feature_key"] not in feature_meta.get("feature_keys",[]):
        raise ValueError(f"特征文件未声明 {settings['feature_key']}")
    for key in ("model","dataset"):
        value=raw.get(key) or feature_meta.get(key)
        if not value:
            raise ValueError(f"缺少 {key} 路径；请在 pipeline 配置中填写")
        candidate=path_from_root(value,root)
        settings[key]=str(candidate) if candidate.exists() else str(value)
        if str(value).startswith(("/","~",".")) and not candidate.exists():
            raise ValueError(f"{key} 路径不存在：{candidate}；请更新 pipeline 配置")
        settings[key+"_revision"]=None if candidate.is_dir() else feature_meta.get(key+"_revision")
    defaults={"retain_logs":"saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json",
        "classifier_model":"models/gibberish-detector", "output_root":"saves/unlearn/representation_npo/seed_runs",
        "manifest_root":"artifacts/representation_batching/seed_runs", "report_dir":"reports/representation_batching"}
    for key,default in defaults.items():
        value=raw.get(key) or default
        settings[key]=str(path_from_root(value,root))
    for key in ("retain_logs",):
        if not Path(settings[key]).is_file() or not read_json(settings[key]):
            raise ValueError(f"缺少有效的 {key}：{settings[key]}")
    for key in ("classifier_model",):
        if not (Path(settings[key]) / "config.json").exists():
            raise ValueError(f"请在配置中将 {key} 指向完整的本地分类器目录：{settings[key]}")
    if Path(settings["model"]).is_dir() and not (Path(settings["model"])/"config.json").exists():
        raise ValueError("Full 模型目录缺少 config.json")
    if Path(settings["dataset"]).is_dir():
        names=("forget05","retain95","forget05_perturbed","retain_perturbed","real_authors_perturbed","world_facts_perturbed")
        missing=[name for name in names if resolve_local_dataset_config(settings["dataset"],name) is None]
        if missing:
            raise ValueError("本地 TOFU 缺少训练/评估数据配置："+", ".join(missing))
    settings["training_gpus"]=str(raw.get("training_gpus","0,1"))
    devices=[x.strip() for x in settings["training_gpus"].split(",")]
    if len(devices)!=2 or len(set(devices))!=2 or any(not x for x in devices):
        raise ValueError("training_gpus 必须是两张不同 GPU，例如 0,1")
    settings["evaluation_gpu"]=str(raw.get("evaluation_gpu","0"))
    if "," in settings["evaluation_gpu"] or not settings["evaluation_gpu"]:
        raise ValueError("evaluation_gpu 必须是一张 GPU")
    settings["retain_size"]=int(raw.get("retain_size",3800))
    settings["evaluation_batch_size"]=int(raw.get("evaluation_batch_size",32))
    settings["learning_rate"]=float(raw.get("learning_rate",2e-5))
    if min(settings["retain_size"],settings["evaluation_batch_size"],settings["learning_rate"])<=0:
        raise ValueError("retain_size、evaluation_batch_size 和 learning_rate 必须为正数")
    settings["feature_sha256"]=actual_hash
    settings["retain_sha256"]=hash_file(settings["retain_logs"])
    settings["scan_roots"]=[str(path_from_root(p,root)) for p in raw.get("scan_roots",["saves/unlearn"])]
    settings["expected_seeds"]=[int(s) for s in raw.get("expected_seeds",[0,1,2])]
    return settings


def fingerprint_payload(settings,root=ROOT,revision=None):
    # Path/config changes cannot silently resume into a different experiment.
    effective={k:v for k,v in settings.items() if k not in ("scan_roots","report_dir","expected_seeds","training_gpus","evaluation_gpu")}
    directories=("src/trainer","src/data","src/model","src/evals","src/representation_batching","configs/model","configs/trainer","configs/accelerate","configs/data","configs/eval","configs/collator")
    files=("src/train.py","src/eval.py","configs/unlearn.yaml","configs/eval.yaml",
                 "configs/experiment/unlearn/tofu/representation_npo.yaml","configs/experiment/eval/tofu/default.yaml",
                 "scripts/representation_batching/build_batch_manifests.py",
                 "scripts/representation_batching/run_npo_representation.sh","scripts/representation_batching/evaluate_run.py")
    if revision:
        names=subprocess.check_output(["git","ls-tree","-r","--name-only",revision],cwd=root,text=True).splitlines()
        names=[name for name in names if name in files or (Path(name).suffix in (".py",".yaml",".json") and any(name.startswith(d+"/") for d in directories))]
        effective["code"]={name:hashlib.sha256(subprocess.check_output(["git","show",f"{revision}:{name}"],cwd=root)).hexdigest() for name in names}
    else:
        sources=set()
        for directory in directories:
            sources.update(p for p in (root/directory).rglob("*") if p.is_file() and p.suffix in (".py",".yaml",".json"))
        sources.update(root/name for name in files if (root/name).exists())
        effective["code"]={str(p.relative_to(root)):hash_file(p) for p in sorted(sources)}
    return effective


def payload_digest(payload,exclude_controls=True):
    payload=dict(payload)
    if exclude_controls:
        payload["code"]={k:v for k,v in payload["code"].items() if k not in CONTROL_FILES}
        if payload["code"].get("src/eval.py")==EVAL_PROVENANCE_FIX_SHA256:
            payload["code"]["src/eval.py"]=LEGACY_EVAL_SHA256
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def fingerprint(settings,root=ROOT):
    return payload_digest(fingerprint_payload(settings,root))


def compatible_legacy_signature(signature,settings,root=ROOT):
    """Accept only a reproducible legacy hash with unchanged numerical sources."""
    current=fingerprint(settings,root)
    for revision in LEGACY_PIPELINE_COMMITS:
        try:
            legacy=fingerprint_payload(settings,root,revision)
            if payload_digest(legacy,False)==signature and payload_digest(legacy)==current:
                return True
        except (OSError,subprocess.CalledProcessError):
            continue
    return False


def commands(settings,seed,arm,run,root=ROOT):
    scripts=root/"scripts/representation_batching"
    manifests=Path(settings["manifest_root"])/f"seed-{seed}"
    build=[sys.executable,str(scripts/"build_batch_manifests.py"),"--features",settings["features"],
        "--feature-key",settings["feature_key"],"--output-dir",str(manifests),"--retain-size",str(settings["retain_size"]),
        "--effective-batch-size","20","--world-size","2","--per-device-batch-size","1",
        "--gradient-accumulation-steps","10","--num-epochs","3","--seed",str(seed)]
    train=["bash",str(scripts/"run_npo_representation.sh"),"--manifest",str(manifests/f"{arm}.jsonl"),
        "--gpu",settings["training_gpus"],"--seed",str(seed),"--model",settings["model"],
        "--dataset",settings["dataset"],"--lr",str(settings["learning_rate"]),"--output-dir",str(run)]
    for key in ("model","dataset"):
        if settings.get(key+"_revision"):
            train += ["--"+key+"-revision",settings[key+"_revision"]]
    evaluate=[sys.executable,str(scripts/"evaluate_run.py"),"--run",str(run),"--retain-logs",settings["retain_logs"],
        "--gpu",settings["evaluation_gpu"],"--dataset",settings["dataset"],"--classifier-model",settings["classifier_model"],
        "--batch-size",str(settings["evaluation_batch_size"])]
    return build,train,evaluate


def write_state(path,state):
    state["updated_at"]=datetime.now(timezone.utc).isoformat()
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temp.replace(path)


def saved_model(run):
    if not (run/"config.json").is_file():
        return False
    for name in ("model.safetensors.index.json","pytorch_model.bin.index.json"):
        if (run/name).is_file():
            try:
                shards=set(read_json(run/name)["weight_map"].values())
                return bool(shards) and all((run/shard).is_file() and (run/shard).stat().st_size>0 for shard in shards)
            except (OSError,KeyError,ValueError):
                return False
    return any((run/name).is_file() and (run/name).stat().st_size>0
               for name in ("model.safetensors","pytorch_model.bin"))


def inspect_run(run):
    try:
        row,_,_=collect_run(run,[],[],include_evaluation=False)
        marker=run/"model_save_complete.json"
        trained=(row["training_status"]=="completed" and saved_model(run)
                 and marker.is_file() and read_json(marker).get("status")=="completed")
    except (OSError,ValueError,KeyError,TypeError):
        return False,False
    # A failed/partially written evaluation must never invalidate saved training.
    try:
        row,_,_=collect_run(run,[],[])
        evaluated=row["eligible"] and all(finite_metric(row.get(m)) for m in METRICS)
    except (OSError,ValueError,KeyError,TypeError):
        evaluated=False
    return trained, trained and evaluated


def finite_metric(value):
    from representation_batching.report import finite
    return finite(value)


def run_command(command,log,root=ROOT):
    print("+ "+shlex.join(command),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a",encoding="utf-8") as handle:
        handle.write("\n+ "+shlex.join(command)+"\n"); handle.flush()
        # Stream output to terminal and per-stage log; subprocess exit code is preserved.
        with subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True) as process:
            try:
                for line in process.stdout:
                    print(line,end="",flush=True); handle.write(line); handle.flush()
                code=process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code,command)


def refresh(settings,seed):
    out=Path(settings["report_dir"])
    roots=list(dict.fromkeys(settings["scan_roots"]+[settings["output_root"]]))
    seeds=sorted(set(settings["expected_seeds"]+[seed]))
    payload=build_report({"roots":roots,"expected_seeds":seeds},out)
    _,warnings=export_comparison(payload["runs"],out,seeds)
    for warning in payload["warnings"]+warnings: print("WARNING:",warning)
    print(f"结果表 / 图：{out/'seed_comparison.html'}",flush=True)


def save_seed_results(seed_root,state):
    """Persist four rows, including pending/failed arms, without model copies."""
    rows=[]
    for arm in ARMS:
        entry=state["arms"].get(arm,{})
        run=Path(entry["run_dir"]) if entry.get("run_dir") else None
        row={"seed":state["seed"],"arm":arm,"status":entry.get("status","pending"),
             **{metric:None for metric in METRICS},"run_dir":str(run) if run else "",
             "summary_path":"","evaluation_path":"","evaluation_status":"missing","evaluation_note":"",
             "error":entry.get("error","")}
        if run:
            summary=run/"evals/TOFU_SUMMARY.json"
            details=run/"evals/TOFU_EVAL.json"
            row["summary_path"]=str(summary) if summary.exists() else ""
            row["evaluation_path"]=str(details) if details.exists() else ""
            try:
                collected,_,_=collect_run(run,[],[])
                row["evaluation_status"]=collected["evaluation_status"]
                row["evaluation_note"]=collected.get("evaluation_note") or ""
                # Failed attempts may contain stale/partial metrics. Only publish
                # metrics from an arm that passed the pipeline's full validation.
                if row["status"]=="completed" and collected["eligible"]:
                    row.update({m:collected.get(m) for m in METRICS})
            except (OSError,ValueError,KeyError,TypeError) as exc:
                row["evaluation_status"]="unreadable_or_incomplete"
                if not row["error"]: row["error"]=str(exc)
        rows.append(row)
    write_state(seed_root/"seed_results.json",{"seed":state["seed"],"status":state.get("status","running"),"runs":rows})
    temp=seed_root/"seed_results.csv.tmp"
    write_csv(temp,rows)
    temp.replace(seed_root/"seed_results.csv")


def describe_incomplete(run,stage):
    try:
        row,_,_=collect_run(run,[],[],include_evaluation=stage=="evaluation")
        details=f"training={row['training_status']}, audit={row['audit']}"
        if stage=="training":
            details+=f", model_files={saved_model(run)}, save_marker={(run/'model_save_complete.json').exists()}"
        else:
            missing=[m for m in METRICS if not finite_metric(row.get(m))]
            details+=f", evaluation={row['evaluation_status']}, missing_metrics={missing}"
        return details
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return f"{type(exc).__name__}: {exc}"


def run_seed(settings,seed,signature,root=ROOT,runner=None,inspector=None,refresher=None):
    runner=runner or (lambda command,log:run_command(command,log,root))
    inspector=inspector or inspect_run
    refresher=refresher or refresh
    seed_root=Path(settings["output_root"])/f"seed-{seed}"
    seed_root.mkdir(parents=True,exist_ok=True)
    with (seed_root/"pipeline.lock").open("a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise ValueError(f"seed {seed} 正在运行，请勿重复启动") from exc
        state_path=seed_root/"pipeline_state.json"
        state=read_json(state_path) if state_path.exists() else {"seed":seed,"signature":signature,"settings":settings,"arms":{}}
        previous_signature=state["signature"]
        if state["signature"]!=signature:
            if signature==fingerprint(settings,root) and compatible_legacy_signature(previous_signature,settings,root):
                state["signature"]=signature
                print("旧版进度已兼容：训练/评估设置与计算代码未改变，保留现有模型与结果。",flush=True)
            else:
                raise ValueError("本 seed 的配置/特征/计算代码与上次不同，已停止自动续跑。请恢复原设置，或在配置中选择新的 output_root 和 manifest_root。")
        manifest_root=Path(settings["manifest_root"])/f"seed-{seed}"
        owner_path=manifest_root/"pipeline_owner.json"
        manifest_files=[manifest_root/f"{arm}.jsonl" for arm in ARMS]
        if any(path.exists() for path in manifest_files):
            if not owner_path.exists() or read_json(owner_path).get("signature") not in (previous_signature,signature):
                raise ValueError(f"清单目录已有其他来源的文件：{manifest_root}；请选择新的 manifest_root，避免覆盖")
        manifest_root.mkdir(parents=True,exist_ok=True)
        write_state(owner_path,{"signature":signature})
        state["status"]="running"
        state.pop("error",None)
        for arm in ARMS:
            state["arms"].setdefault(arm,{"attempts":[],"status":"pending"})
        write_state(state_path,state)
        save_seed_results(seed_root,state)
        try:
            if not all(path.exists() for path in manifest_files):
                runner(commands(settings,seed,"R",seed_root/"R/attempt-0001",root)[0],seed_root/"logs/manifests.log")
            failures=[]
            for index,arm in enumerate(ARMS,1):
                entry=state["arms"][arm]
                entry.pop("error",None)
                stage="training"
                try:
                    current=Path(entry["run_dir"]) if entry.get("run_dir") else None
                    trained,evaluated=inspector(current) if current else (False,False)
                    if not trained:
                        attempt=len(entry["attempts"])+1
                        current=seed_root/arm/f"attempt-{attempt:04d}"
                        while current.exists():
                            attempt+=1; current=seed_root/arm/f"attempt-{attempt:04d}"
                        entry["run_dir"]=str(current); entry["attempts"].append(str(current)); entry["status"]="training"
                        write_state(state_path,state)
                        save_seed_results(seed_root,state)
                        print(f"[seed {seed}] [{index}/4] {arm}: 开始训练 → {current}",flush=True)
                        runner(commands(settings,seed,arm,current,root)[1],seed_root/f"logs/{arm}-attempt-{attempt:04d}-train.log")
                        trained,evaluated=inspector(current)
                        if not trained:
                            raise ValueError(f"训练命令返回但完成检查未通过：{describe_incomplete(current,'training')}；目录：{current}")
                    else:
                        print(f"[seed {seed}] [{index}/4] {arm}: 跳过已完成训练",flush=True)
                    stage="evaluation"
                    if not evaluated:
                        entry["status"]="evaluating"; write_state(state_path,state)
                        save_seed_results(seed_root,state)
                        print(f"[seed {seed}] [{index}/4] {arm}: 开始评估 → {current/'evals'}",flush=True)
                        runner(commands(settings,seed,arm,current,root)[2],seed_root/f"logs/{arm}-eval.log")
                        if not inspector(current)[1]:
                            raise ValueError(f"评估命令返回但完成检查未通过：{describe_incomplete(current,'evaluation')}；目录：{current}")
                    else:
                        print(f"[seed {seed}] [{index}/4] {arm}: 跳过已完成评估",flush=True)
                        try:
                            note=collect_run(current,[],[])[0].get("evaluation_note")
                            if note: print(f"[seed {seed}] {arm}: {note}",flush=True)
                        except (OSError,ValueError,KeyError,TypeError):
                            pass
                    entry["status"]="completed"
                except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as exc:
                    entry["status"]=stage+"_failed"
                    entry["error"]=f"{type(exc).__name__}: {exc}"
                    failures.append(f"{arm}: {entry['error']}")
                    print(f"[seed {seed}] [{index}/4] {arm}: {stage} 失败；已记录，将继续其余组。\n{entry['error']}",file=sys.stderr,flush=True)
                except (KeyboardInterrupt,SystemExit):
                    entry["status"]="interrupted"
                    entry["error"]=f"Interrupted during {stage}"
                    raise
                finally:
                    write_state(state_path,state)
                    save_seed_results(seed_root,state)
            if failures:
                state["status"]="partial_failed"
                raise ValueError("四组已全部尝试，以下组未完成：\n"+"\n".join(failures)+f"\n修复原因后重跑同一 seed；进度与结果：{seed_root/'seed_results.csv'}")
            state["status"]="completed"
        except BaseException as exc:
            if state["status"]!="partial_failed": state["status"]="interrupted_or_failed"
            state["error"]=f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_state(state_path,state)
            save_seed_results(seed_root,state)
            try: refresher(settings,seed)
            except Exception as exc: print(f"WARNING: 汇总暂未生成：{exc}；可单独运行 compare_seeds.py。",file=sys.stderr)
            print(f"seed {seed} 四组状态："+", ".join(f"{a}={state['arms'][a]['status']}" for a in ARMS),flush=True)
            print(f"本 seed 结果：{seed_root/'seed_results.csv'}",flush=True)
    return state


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Pipeline interrupted")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed",type=int)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/analysis/representation_pipeline.json")
    parser.add_argument("--dry-run",action="store_true",help="Validate input paths and print commands without launching jobs or writing state")
    args=parser.parse_args()
    if args.seed<0: parser.error("seed 必须是非负整数")
    try:
        settings=load_settings(args.config)
        print(f"seed={args.seed}; Full={settings['model']}; TOFU={settings['dataset']}",flush=True)
        if args.dry_run:
            for index,arm in enumerate(ARMS):
                run=Path(settings["output_root"])/f"seed-{args.seed}"/arm/"attempt-0001"
                build,train,evaluate=commands(settings,args.seed,arm,run)
                for command in ([build] if index==0 else [])+[train,evaluate]: print(shlex.join(command))
            print("最后自动刷新 reports 中的总览、seed_comparison.csv 和 PNG/PDF/SVG。")
            return
        missing=[name for name in ("numpy","torch","transformers","accelerate","hydra") if importlib.util.find_spec(name) is None]
        if missing: raise ValueError("当前 Python 环境缺少："+", ".join(missing)+"；请先激活项目训练环境")
        run_seed(settings,args.seed,fingerprint(settings))
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:
        parser.exit(1,f"ERROR: {exc}\n")


if __name__=="__main__":
    main()
