"""Evaluate retained baseline checkpoints on a matched 50/50 test control.

This is a successor evaluation protocol: it does not retrain models and it
does not overwrite the original results_seed*.json files.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from neuralop import Trainer
from neuralop.losses import LpLoss
from torch.utils.data import DataLoader

from data import get_dataloaders
from model import build_model


def evaluate_seed(cfg, checkpoint, device):
    matched_cfg = copy.deepcopy(cfg)
    matched_cfg["data"]["n_tests"] = [50]
    _, raw_test_loaders, data_processor = get_dataloaders(matched_cfg)
    # Keep the evaluation path identical to train.py while avoiding worker
    # processes in this standalone control evaluator.
    test_loaders = {
        resolution: DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for resolution, loader in raw_test_loaders.items()
    }
    data_processor = data_processor.to(device)

    model = build_model(cfg).to(device)
    # These checkpoints were generated locally by this repo's train.py.
    # Explicitly retain the legacy loading behavior required by this torch
    # build for the serialized FNO state dict.
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)
    )
    trainer = Trainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=5,
        use_distributed=False,
        verbose=False,
    )
    metrics = trainer.evaluate_all(
        epoch=0,
        eval_losses={"l2": LpLoss(d=2, p=2)},
        test_loaders=test_loaders,
        eval_modes={},
    )
    return {resolution: value.item() for resolution, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="outputs/matched_control.json")
    parser.add_argument("--checkpoints", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_seed = {}
    for seed in (0, 1, 2):
        checkpoint = Path(args.checkpoints) / f"fno_darcy_seed{seed}.pt"
        if not checkpoint.exists():
            raise SystemExit(f"Missing retained checkpoint: {checkpoint}")
        per_seed[str(seed)] = evaluate_seed(cfg, checkpoint, device)

    in_dist = [v["16_l2"] for v in per_seed.values()]
    ood = [v["32_l2"] for v in per_seed.values()]
    mean_in = sum(in_dist) / len(in_dist)
    mean_ood = sum(ood) / len(ood)
    result = {
        "protocol": "matched_test_n50_successor_v1",
        "training_config_unchanged": True,
        "training_reused_retained_checkpoints": True,
        "test_samples_per_resolution": 50,
        "per_seed": per_seed,
        "aggregate": {
            "mean_relative_l2_res16": mean_in,
            "mean_relative_l2_res32": mean_ood,
            "resolution_gap": mean_ood - mean_in,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
