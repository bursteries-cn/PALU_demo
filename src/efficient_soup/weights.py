"""Read immutable safetensors ingredients and merge one tensor at a time."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .search import coefficients


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Ingredient:
    def __init__(self, path, stack):
        self.path = Path(path).expanduser().resolve()
        if (self.path / "adapter_config.json").exists():
            raise ValueError("Merge LoRA adapters into their common base before souping")
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text())["weight_map"]
        elif (self.path / "model.safetensors").is_file():
            mapping = None
        else:
            raise ValueError(f"Full safetensors weights missing: {self.path}; train with KEEP_MODEL=1")
        names = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
        self.handles, self.stats = {}, {}
        for name in names:
            file = self.path / name
            if Path(name).name != name:
                raise ValueError("Shard names must be filenames")
            self.handles[name] = stack.enter_context(safe_open(str(file), framework="pt", device="cpu"))
            stat = file.stat()
            self.stats[name] = (stat.st_size, stat.st_mtime_ns)
        self.mapping = mapping or {key: names[0] for key in self.handles[names[0]].keys()}
        actual = {key: name for name, handle in self.handles.items() for key in handle.keys()}
        if self.mapping != actual:
            raise ValueError(f"Incomplete or inconsistent safetensors index: {self.path}")

    def tensor(self, key):
        return self.handles[self.mapping[key]].get_tensor(key)

    def assert_unchanged(self):
        for name, expected in self.stats.items():
            stat = (self.path / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError(f"Ingredient changed during search: {self.path / name}")

    def fingerprint(self):
        files = sorted(self.mapping.values())
        files = sorted(set(files) | {"config.json"})
        return {"path": str(self.path), "sha256": {name: sha256(self.path / name) for name in files}}


class WeightMixer:
    def __init__(self, model, paths):
        if len(paths) != 3 or len({Path(p).resolve() for p in paths}) != 3:
            raise ValueError("Provide three distinct Full/candidate checkpoint directories")
        self.stack = ExitStack()
        try:
            self.sources = [Ingredient(path, self.stack) for path in paths]
            self.targets = model.state_dict(keep_vars=True)
            # HF safetensors may omit one name of a tied parameter.
            aliases = {}
            for key, tensor in self.targets.items():
                aliases.setdefault((tensor.data_ptr(), tuple(tensor.shape)), []).append(key)
            self.groups = list(aliases.values())
            expected = set(self.targets)
            self.keys = []
            for source in self.sources:
                if set(source.mapping) - expected:
                    raise ValueError("Checkpoint contains unexpected state keys")
                chosen = []
                for group in self.groups:
                    available = [key for key in group if key in source.mapping]
                    if not available:
                        raise ValueError(f"Missing state tensor: {group}")
                    tensor = source.tensor(available[0])
                    target = self.targets[group[0]]
                    if tensor.shape != target.shape or tensor.is_floating_point() != target.is_floating_point():
                        raise ValueError(f"Incompatible tensor: {available[0]}")
                    if any(not torch.equal(tensor, source.tensor(key)) for key in available[1:]):
                        raise ValueError("Checkpoint disagrees with model's tied-weight configuration")
                    chosen.append(available[0])
                self.keys.append(chosen)
        except Exception:
            self.close()
            raise

    def close(self):
        self.stack.close()

    @torch.no_grad()
    def apply(self, weights):
        weights = coefficients(weights)
        for source in self.sources:
            source.assert_unchanged()
        for index, group in enumerate(self.groups):
            target = self.targets[group[0]]
            if not target.is_floating_point():
                value = self.sources[0].tensor(self.keys[0][index])
                if any(not torch.equal(value, s.tensor(keys[index]))
                       for s, keys in zip(self.sources[1:], self.keys[1:])):
                    raise ValueError(f"Nonfloating buffers differ: {group[0]}")
            else:
                value = torch.zeros(target.shape, dtype=torch.float32, device="cpu")
                for weight, source, keys in zip(weights, self.sources, self.keys):
                    if weight:
                        value.add_(source.tensor(keys[index]).float(), alpha=weight)
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite merged weights: {group[0]}")
            target.copy_(value)
