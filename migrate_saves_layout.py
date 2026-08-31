"""在 saves/unlearn 的模型层下面插入一层日期目录，并把已有实验结果归入对应日期。

旧结构：saves/unlearn/<benchmark>/<split>/<model>/<trainer>/<run>/checkpoint-N/...
新结构：saves/unlearn/<benchmark>/<split>/<model>/<YYYY-MM-DD>/<trainer>/<run>/checkpoint-N/...

日期从 run 目录名里的 `_dayMMDD_timeHHMMSS` 解析；年份由目录 mtime 推断
（跨年的情况下取不晚于 mtime 的那个年份）。名字里没有日期时退回用 mtime。

用法示例：
    python migrate_saves_layout.py                 # 只打印迁移计划，不动文件
    python migrate_saves_layout.py --apply         # 真正执行
    python migrate_saves_layout.py --apply --min-idle-minutes 30
"""

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DAY_TIME_RE = re.compile(r"_day(?P<day>\d{4})_time(?P<time>\d{6})")


def is_run_dir(path):
    """run 目录的判据：名字带 day/time 后缀，或里面有 checkpoint-N 子目录。

    前者能把只留下 PALU.log 的失败 run 也一起归位，后者兼容手工改过名的目录。
    """
    if DAY_TIME_RE.search(path.name):
        return True
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in path.iterdir())


def dir_mtime(path):
    """取目录自身和一层子项里最早的 mtime，尽量贴近实验开始时间。"""
    stamps = [path.stat().st_mtime]
    stamps += [p.stat().st_mtime for p in path.iterdir()]
    return datetime.fromtimestamp(min(stamps))


def run_date(run_dir):
    """返回 (YYYY-MM-DD, 依据说明)。"""
    mtime = dir_mtime(run_dir)
    m = DAY_TIME_RE.search(run_dir.name)
    if m is None:
        return mtime.strftime("%Y-%m-%d"), "mtime（目录名里没有 day/time）"

    day, tm = m.group("day"), m.group("time")
    # 目录名里只有 MMDD，年份候选取 mtime 当年和上一年，选不晚于 mtime 的最近一个
    best = None
    for year in (mtime.year, mtime.year - 1):
        try:
            cand = datetime.strptime(f"{year}{day}{tm}", "%Y%m%d%H%M%S")
        except ValueError:
            continue  # 2 月 29 日之类在非闰年不存在
        if cand <= mtime and (best is None or cand > best):
            best = cand
    if best is None:
        return mtime.strftime("%Y-%m-%d"), "mtime（目录名日期晚于 mtime，可能是手工改名）"
    return best.strftime("%Y-%m-%d"), "目录名 day/time"


def plan(root, min_idle_minutes):
    """扫描 root 下所有 run 目录，返回 (迁移计划, 已就位的数量, 被跳过的活跃目录)。"""
    moves, already, busy = [], [], []
    now = datetime.now().timestamp()

    for run_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        try:
            if not is_run_dir(run_dir):
                continue
        except OSError:
            continue

        # run_dir = <root>/<benchmark>/<split>/<model>/[<date>/]<trainer>/<run>
        trainer_dir = run_dir.parent
        model_dir = trainer_dir.parent
        if DATE_DIR_RE.match(model_dir.name):
            already.append(run_dir)
            continue

        idle_min = (now - max(p.stat().st_mtime for p in [run_dir, *run_dir.iterdir()])) / 60
        if idle_min < min_idle_minutes:
            busy.append((run_dir, idle_min))
            continue

        date, basis = run_date(run_dir)
        dest = model_dir / date / trainer_dir.name / run_dir.name
        moves.append((run_dir, dest, date, basis))

    return moves, already, busy


def prune_empty(path, stop_at):
    """迁移后清掉一路空下来的旧 trainer 目录。"""
    while path != stop_at and path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        path = path.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="saves/unlearn")
    ap.add_argument("--apply", action="store_true", help="不加这个参数只打印计划")
    ap.add_argument(
        "--min-idle-minutes",
        type=float,
        default=5.0,
        help="最近还有写入的 run 目录会被跳过，避免动到正在跑的实验",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"目录不存在：{root}")

    moves, already, busy = plan(root, args.min_idle_minutes)

    if already:
        print(f"已经在日期层下、无需处理：{len(already)} 个 run")
    if busy:
        print(f"\n跳过 {len(busy)} 个最近仍有写入的 run（可能正在训练）：")
        for run_dir, idle in busy:
            print(f"  {idle:5.1f} 分钟前刚写过  {run_dir}")
    if not moves:
        print("\n没有需要迁移的 run 目录。")
        return

    by_date = defaultdict(list)
    for run_dir, dest, date, basis in moves:
        by_date[date].append((run_dir, dest, basis))

    print(f"\n待迁移 {len(moves)} 个 run，落到 {len(by_date)} 个日期：")
    for date in sorted(by_date):
        items = by_date[date]
        bases = {b for _, _, b in items}
        print(f"  {date}  {len(items):>4} 个 run   依据：{', '.join(sorted(bases))}")

    sample_src, sample_dst = moves[0][0], moves[0][1]
    print("\n示例：")
    print(f"  从  {sample_src}")
    print(f"  到  {sample_dst}")

    conflicts = [(s, d) for s, d, _, _ in moves if d.exists()]
    if conflicts:
        print(f"\n目标已存在，无法迁移（{len(conflicts)} 个），请先手工处理：")
        for s, d in conflicts:
            print(f"  {s}\n  -> {d}")
        raise SystemExit(1)

    if not args.apply:
        print("\n这是 dry-run。确认无误后加 --apply 执行。")
        return

    print()
    for i, (src, dest, _, _) in enumerate(moves, 1):
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dest)  # 同一文件系统内是重命名，不会真的搬数据
        prune_empty(src.parent, root)
        print(f"[{i}/{len(moves)}] {src.name} -> {dest.parent}")

    print(f"\n完成，共迁移 {len(moves)} 个 run 目录。")


if __name__ == "__main__":
    main()
