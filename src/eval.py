import hydra
import hashlib
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from evals import get_evaluators
from model import get_model
from trainer.utils import seed_everything
from representation_batching.evaluation_config import set_tofu_dataset_paths, validate_local_tofu_files


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    dataset_override = cfg.get("tofu_dataset_path")
    if dataset_override:
        evaluation = OmegaConf.to_container(cfg.eval, resolve=True)
        set_tofu_dataset_paths(evaluation, dataset_override)
        validate_local_tofu_files(evaluation, dataset_override)
        cfg.eval = OmegaConf.create(evaluation)
    # get_model consumes model_args.path/dtype in place. Snapshot the resolved
    # request before loading, so provenance retains the actual model and precision.
    requested_config = OmegaConf.to_container(cfg, resolve=True)
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        output_dir = Path(str(eval_cfgs[evaluator_name].output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        source_root = Path(__file__).resolve().parent
        sources = sorted((source_root / "evals").rglob("*.py"))
        sources += sorted((source_root / "data").rglob("*.py"))
        sources += sorted((source_root / "model").rglob("*.py"))
        provenance = {
            "config": requested_config,
            "code_sha256": {
                str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
            "status": "started",
        }
        reference = eval_cfgs[evaluator_name].get("retain_logs_path")
        if reference and Path(str(reference)).is_file():
            provenance["reference_sha256"] = hashlib.sha256(Path(str(reference)).read_bytes()).hexdigest()
        provenance_path = output_dir / "evaluation_provenance.json"
        try:
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        cached = any(output_dir.glob("*_EVAL.json")) and not eval_cfgs[evaluator_name].overwrite
        verified_cache = (
            previous.get("status") == "completed"
            and previous.get("config") == provenance["config"]
            and previous.get("code_sha256") == provenance["code_sha256"]
            and previous.get("reference_sha256") == provenance.get("reference_sha256")
        )
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)
        provenance["status"] = "completed_unverified_cache" if cached and not verified_cache else "completed"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
