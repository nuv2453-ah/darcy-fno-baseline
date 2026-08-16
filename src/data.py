"""
Dataset loading for the 2D Darcy flow baseline.

Uses the official small Darcy flow dataset shipped by the `neuraloperator`
library (same data used in the original FNO paper, Li et al. 2020).
Data is hosted on Zenodo (https://zenodo.org/records/12784353) and is
downloaded automatically on first run -- this requires normal internet
access, so run this on your own machine / Colab rather than a sandboxed
environment with restricted network egress.
"""

from neuralop.data.datasets import load_darcy_flow_small


def get_dataloaders(cfg):
    """Build train/test dataloaders from the fixed config.

    Train resolution is fixed at 16x16 by the dataset loader itself.
    Test resolutions of [16, 32] give us both an in-distribution test
    set and a resolution-generalization test set for free.
    """
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=cfg["data"]["n_train"],
        n_tests=cfg["data"]["n_tests"] * len(cfg["data"]["test_resolutions"]),
        batch_size=cfg["data"]["batch_size"],
        test_batch_sizes=cfg["data"]["test_batch_sizes"] * len(cfg["data"]["test_resolutions"]),
        test_resolutions=cfg["data"]["test_resolutions"],
    )
    return train_loader, test_loaders, data_processor
