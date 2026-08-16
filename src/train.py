"""
Train the FNO baseline on 2D Darcy flow.

Usage:
    python src/train.py --config configs/default.yaml --seed 0
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from neuralop import Trainer
from neuralop.losses import LpLoss
from neuralop.training import AdamW

from data import get_dataloaders
from model import build_model


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    set_seed(seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        # MPS produces NaN losses for FNO's complex-valued spectral (FFT)
        # layers on this torch build (confirmed by direct comparison), so
        # we pin to CPU on Apple Silicon rather than use torch.backends.mps.
        device = "cpu"

    train_loader, test_loaders, data_processor = get_dataloaders(cfg)
    data_processor = data_processor.to(device)

    model = build_model(cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["scheduler_step_size"],
        gamma=cfg["train"]["scheduler_gamma"],
    )

    l2loss = LpLoss(d=2, p=2)

    trainer = Trainer(
        model=model,
        n_epochs=cfg["train"]["n_epochs"],
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=5,
        use_distributed=False,
        verbose=True,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=l2loss,
        eval_losses={"l2": l2loss},
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"fno_darcy_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Final eval pass — reuse Trainer's own evaluation path instead of
    # hand-rolling it (the hand-rolled version divided by batch count, not
    # sample count, and inflated the reported error roughly 50x).
    eval_metrics = trainer.evaluate_all(
        epoch=cfg["train"]["n_epochs"] - 1,
        eval_losses={"l2": l2loss},
        test_loaders=test_loaders,
        eval_modes={},
    )
    print("eval_metrics keys:", list(eval_metrics.keys()))  # sanity check

    results = {"seed": seed}
    for res in test_loaders:
        val = eval_metrics[f"{res}_l2"]
        results[f"relative_l2_res{res}"] = val.item() if hasattr(val, "item") else val

    with open(out_dir / f"results_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
