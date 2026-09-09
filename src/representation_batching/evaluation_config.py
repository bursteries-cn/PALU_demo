"""Small, dependency-free helpers for consistent offline TOFU evaluation."""
from pathlib import Path
from representation_batching.local_dataset import resolve_local_dataset_config


def set_tofu_dataset_paths(node, dataset_path):
    """Override only TOFU dataset entries; preserve config name and all metric settings."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.startswith("TOFU_") and isinstance(value, dict):
                args = value.get("args", {}).get("hf_args")
                if args is not None:
                    args["path"] = str(dataset_path)
            set_tofu_dataset_paths(value, dataset_path)
    elif isinstance(node, list):
        for value in node:
            set_tofu_dataset_paths(value, dataset_path)


def validate_local_tofu_files(node, dataset_path):
    if not Path(dataset_path).is_dir():
        return
    names = set()
    def visit(value):
        if isinstance(value, dict):
            for key, entry in value.items():
                if key.startswith("TOFU_") and isinstance(entry, dict):
                    name = entry.get("args", {}).get("hf_args", {}).get("name")
                    if name:
                        names.add(name)
                visit(entry)
        elif isinstance(value, list):
            for entry in value:
                visit(entry)
    visit(node)
    missing = [name for name in sorted(names) if resolve_local_dataset_config(dataset_path, name) is None]
    if missing:
        raise ValueError(f"Local TOFU evaluation files missing in {dataset_path}: {', '.join(missing)}")
