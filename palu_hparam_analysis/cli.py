from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .report import run_analysis
from .schema import load_analysis_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record sparse PALU hyperparameter experiments, summarize promising "
            "observed values, and propose a few untested one-step neighbors."
        )
    )
    parser.add_argument("--root", default="saves/unlearn", help="result tree to scan")
    parser.add_argument(
        "--config",
        default="configs/analysis/palu_hparams.yaml",
        help="analysis configuration (JSON-compatible YAML by default)",
    )
    parser.add_argument(
        "--out", default="reports/hparam_analysis/latest", help="output report directory"
    )
    parser.add_argument("--model", default=None, help="override model filter")
    parser.add_argument("--split", default=None, help="override forget split filter")
    parser.add_argument("--date", default=None, help="optional YYYY-MM-DD run-date filter")
    parser.add_argument(
        "--checkpoint-policy", choices=("epoch", "last"), default=None
    )
    parser.add_argument("--epoch", type=float, default=None, help="fixed epoch to compare")
    parser.add_argument(
        "--allow-mixed-protocols",
        action="store_true",
        help="analyze protocol cohorts together (unsafe unless differences are intentional)",
    )
    return parser


def _override(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.model is not None:
        config["filters"]["model"] = args.model
    if args.split is not None:
        config["filters"]["split"] = args.split
    if args.date is not None:
        config["filters"]["date"] = args.date
    if args.checkpoint_policy is not None:
        config["checkpoint"]["policy"] = args.checkpoint_policy
    if args.epoch is not None:
        config["checkpoint"]["epoch"] = args.epoch
    if args.allow_mixed_protocols:
        config["protocol"]["allow_mixed"] = True


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    result_root = Path(args.root)
    if not result_root.is_absolute():
        result_root = repo_root / result_root
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    config = load_analysis_config(config_path)
    _override(config, args)
    manifest = run_analysis(result_root, config, output_dir, repo_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
