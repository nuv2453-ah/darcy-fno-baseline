"""Model construction for the baseline. Kept as a thin wrapper so swapping
in DeepONet or another operator model later only touches this file."""

from neuralop.models import FNO


def build_model(cfg):
    m = cfg["model"]
    if m["name"] != "FNO":
        raise NotImplementedError(
            f"Only FNO is wired up in this baseline; got {m['name']}. "
            "Add a branch here when you move past the baseline pass."
        )
    return FNO(
        n_modes=tuple(m["n_modes"]),
        hidden_channels=m["hidden_channels"],
        n_layers=m["n_layers"],
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
    )
