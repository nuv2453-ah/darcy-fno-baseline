# Reproducibility and evidence note

This note records the original baseline without rewriting its outputs. The
baseline evidence was introduced in commit
`04768597cb03cd83e0cd33a8467378a267df40ea`; the recorded repository head at
the time of this note is `82d12ea0b7d4a549c98372c17a32ff907fcaf627`.

## Runtime used for the successful CPU runs

- Hardware: Apple MacBook Pro (Model14,7), Apple M2, 8 GB RAM, 8 CPU cores.
- OS/platform: macOS 26.5.1, arm64.
- Python: 3.11.15.
- PyTorch: 2.13.0.
- neuraloperator: 2.0.0.
- NumPy: 2.4.6; SciPy: 1.17.1; PyYAML: 6.0.3; h5py: 3.16.0.
- tensorly: 0.9.0; tensorly-torch: 0.5.0; matplotlib: 3.11.1;
  wandb: 0.28.2.
- CUDA was unavailable. MPS was available, but the original MPS run produced
  NaN losses in the FNO spectral layers; the successful runs therefore used
  `device="cpu"`.

## Exact baseline commands

From the repository root, with `venv` activated:

```bash
python src/train.py --config configs/default.yaml --seed 0
python src/train.py --config configs/default.yaml --seed 1
python src/train.py --config configs/default.yaml --seed 2
python src/eval.py --config configs/default.yaml --out outputs
```

Seeds 1 and 2 plus aggregation were also run as one chained shell command;
the individual commands above are the exact command units.

## Observed dataset counts

The training file contains 1,000 examples at 16x16. The baseline loader was
called with `n_tests=[200, 200]` after `src/data.py` expanded the configured
`n_tests=[200]` across the two resolutions. The observed counts were:

| Split | Resolution | Requested | Observed |
|---|---:|---:|---:|
| train | 16x16 | 1,000 | 1,000 |
| test | 16x16 | 200 | 50 |
| test | 32x32 | 200 | 200 |

The underlying files explain the mismatch: `darcy_test_16.pt` contains 50
examples, while `darcy_test_32.pt` contains 1,000 and is sliced to the first
200 by the loader. `PTDataset` uses `slice(0, n_test)`; it cannot synthesize
additional 16x16 examples. The source is the small Darcy Flow dataset from
[Zenodo record 12784353](https://zenodo.org/records/12784353), accessed via
`neuraloperator==2.0.0` and its `load_darcy_flow_small` / `DarcyDataset`
implementation. The seven-paper mini literature review remains in
`project_brief.md`.

## Retained baseline artifacts

SHA-256 hashes of the original result JSON files:

```text
4503eb29750bb7e235202c550b6a7f2103a9758613b7dcbb57230b062e1e3b6d  outputs/results_seed0.json
c348699b85970af17acf71eff9927347452f4a40cabb875b8c9ecfc03ea2a80f  outputs/results_seed1.json
58a608b9d452d1a91cb5e1fbeb0d79cdf0a1d46223f28ec946acce34cc566bb7  outputs/results_seed2.json
```

The original verdict remains **INCONCLUSIVE** under the predeclared rule:
success requires mean in-distribution relative L2 below 0.05; failure is
above 0.15; and high seed variance is separately inconclusive.

## Post-hoc exploratory matched-count control

`src/matched_eval.py` defines `matched_test_n50_successor_v1`. It reuses the
retained checkpoints, unchanged training configuration, unchanged model, and
the exact `Trainer`/`LpLoss` evaluation path, while evaluating 50 examples at
both resolutions. It does not overwrite the original JSON files. This control
was executed after the original 50-vs-200 outcome was known, so it is
post-hoc exploratory evidence, not a pre-registered confirmatory successor.

Command:

```bash
python src/matched_eval.py --config configs/default.yaml \
  --out outputs/matched_control_v1.json
```

Results:

| Seed | 16x16 L2 (50) | 32x32 L2 (50) | Gap |
|---:|---:|---:|---:|
| 0 | 0.1296 | 0.1842 | 0.0546 |
| 1 | 0.1133 | 0.1932 | 0.0800 |
| 2 | 0.1016 | 0.1678 | 0.0662 |
| Mean | 0.1148 | 0.1817 | 0.0669 |

The matched control leaves the formal baseline verdict **INCONCLUSIVE**. The
32x32 error remains higher than 16x16 after matching the evaluation count,
which is consistent with a possible resolution-generalization effect, but
the original 50-vs-200 comparison had a sample-count/protocol confound. This
control reduces that confound; it does not establish causality or justify
retuning.

The matched-control JSON has SHA-256:

```text
4b3a29d9e34a61eb7818777b7f6e6c5f51e7f59f8bab77d6c76b76b480fb1a80  outputs/matched_control_v1.json
```

Any future 200-vs-200 run or revised cross-resolution transfer criterion must
be a separately versioned confirmatory successor protocol, with data
cardinalities/provenance, hypothesis, metrics, and decision rule frozen before
execution. The original v1 and this exploratory control remain immutable
evidence packages.
