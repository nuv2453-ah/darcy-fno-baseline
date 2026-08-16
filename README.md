# 2D Darcy Flow — FNO Baseline

Physics-Informed AI & Neural Operators track, BU1LD Research Program.

Minimal, reproducible FNO baseline for 2D Darcy flow, built on the official
[`neuraloperator`](https://github.com/neuraloperator/neuraloperator) library.
See `project_brief.md` for the research question, hypothesis, literature
review, and success criteria.

## Setup

```bash
pip install -r requirements.txt
```

Data (from [Zenodo](https://zenodo.org/records/12784353), same set used in
the original FNO paper) downloads automatically on first run — needs normal
internet access, so run locally or on Colab rather than a network-restricted
sandbox.

## Run

```bash
# Train + evaluate a single seed
python src/train.py --config configs/default.yaml --seed 0

# Repeat for seeds 1, 2 (per the protocol in configs/default.yaml)
python src/train.py --config configs/default.yaml --seed 1
python src/train.py --config configs/default.yaml --seed 2

# Aggregate across seeds and apply the pre-registered thresholds
python src/eval.py --config configs/default.yaml --out outputs
```

## Structure

```
configs/default.yaml   # fixed experimental protocol — data, model, thresholds
src/data.py             # dataset loading
src/model.py             # FNO construction
src/train.py             # train + per-seed eval, writes outputs/results_seed*.json
src/eval.py               # cross-seed aggregation + success/failure/inconclusive verdict
project_brief.md          # one-page brief for review
```

## Result

The three-seed baseline has been run end-to-end on CPU because the available
Apple MPS backend produced NaNs in the FNO spectral layers. Results:

| Seed | 16x16 relative L2 | 32x32 relative L2 |
|---:|---:|---:|
| 0 | 0.1296 | 0.1908 |
| 1 | 0.1133 | 0.1892 |
| 2 | 0.1016 | 0.1682 |
| Mean | 0.1148 | 0.1827 |

The aggregate verdict is **INCONCLUSIVE (between thresholds)**. The mean
resolution-generalization gap is 0.0679. The loader reports 50 samples at
16x16 and 200 at 32x32 despite the requested 200 per resolution; this is
recorded as a protocol limitation rather than silently corrected.

Per-seed JSON results are in `outputs/results_seed*.json`. Checkpoints are
intentionally ignored by Git because they are large generated artifacts.
