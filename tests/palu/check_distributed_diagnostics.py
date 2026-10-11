"""Two CPU ranks: verify diagnostic collectives, padding, and rank-zero history."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/palu"), str(Path(__file__).parent)]
from test_diagnostics import fixture, cache_fixture
from trainer.unlearn.palu import PALU


def worker(rank, directory):
    directory = Path(directory)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29617")
    dist.init_process_group("gloo", init_method="file://" + str(directory / "rendezvous"),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    model, tokenizer, dataset, collator = fixture()
    options = json.loads((directory / "options.json").read_text())
    args = TrainingArguments(output_dir=str(directory / "run"), use_cpu=True, max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, optim="adamw_torch",
        learning_rate=.001, report_to=[], save_strategy="no", eval_strategy="no", logging_strategy="no",
        remove_unused_columns=False, disable_tqdm=True, seed=11, data_seed=11)
    trainer = PALU(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset,
        data_collator=collator, top_k=3, first_n=1, alpha=.2, track_loss_components=True, diagnostics=options)
    trainer.train()
    torch.save(model.state_dict(), directory / f"parameters_rank{rank}.pt")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="palu-gloo-"))
    if args.output:
        directory.mkdir(parents=True, exist_ok=False)
    initial, tokenizer, dataset, collator = fixture()
    options = cache_fixture(directory / "reference.pt", initial, tokenizer, dataset, collator)
    (directory / "options.json").write_text(json.dumps(options))
    mp.spawn(worker, args=(str(directory),), nprocs=2, join=True)
    rows = [json.loads(line) for line in (directory / "run/probe_history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert all(row["forget_count"] == 3 and row["retain_count"] == 2 for row in rows)
    samples = [json.loads(line) for line in (directory / "run/probe_samples.jsonl").read_text().splitlines()]
    assert len(samples) == 15
    first = torch.load(directory / "parameters_rank0.pt", weights_only=True)
    second = torch.load(directory / "parameters_rank1.pt", weights_only=True)
    assert all(torch.equal(value, second[key]) for key, value in first.items())
    print(f"Two CPU ranks passed: identical parameters, one history, padding ignored. Output: {directory}")


if __name__ == "__main__":
    main()
