# P1: Paired 200-sample constructed-resolution confirmatory protocol

**Status:** Pre-execution frozen, pending review. No outcome-producing run is authorized by this document.

**Parent evidence:** Frozen v1 and the post-hoc exploratory control remain unchanged at commit `5c81e262624acd2b77a25bc15500d3b476840a19`. P1 must write only to `outputs/p1_paired_200/` and must not modify any `outputs/results_seed*.json` or `outputs/matched_control_v1.json` artifact.

## Question

For an FNO trained from scratch on the fixed 1,000 native 16x16 Darcy samples, does performance on a paired, held-out 200-sample set deteriorate at 32x32 relative to a deterministically constructed 16x16 version of those same examples?

This is a new constructed-resolution protocol. It does **not** claim that the source provides 200 native 16x16 test examples, and it does not change the meaning of v1 or its exploratory 50-vs-50 control.

## Fixed data and provenance

- Dataset: Small Darcy Flow, Zenodo record `12784353`, loaded through `neuraloperator==2.0.0`.
- Train set: `darcy_train_16.pt`, first 1,000 native 16x16 samples; no validation-based selection or hyperparameter tuning.
- Test source: `darcy_test_32.pt`, SHA-256 `f9fbe30065908327f132425d760d573ed7ce29cb2a54cfdb3fcefd3d80d7ac83`.
- Paired test IDs: Python half-open slice `[200:400]`, yielding 200 examples. These are disjoint from v1's first-200 32x32 test slice.
- 32x32 arm: native `x` and `y` fields for IDs `[200:400]`.
- 16x16 arm: apply `torch.nn.functional.interpolate` independently to each selected `x` and `y` field with `size=(16, 16)`, `mode="bilinear"`, `align_corners=False`, and `antialias=False`.

The two arms therefore have identical sample identities and cardinality. The interpolation rule is part of the estimand; it must not be replaced by a native-16 file, a different interpolation mode, or a different index range after results are seen.

## Fixed model and execution

Use the FNO and training choices in `configs/p1_paired_200.yaml` without modification: four Fourier layers, 32 hidden channels, 16 modes per spatial dimension, AdamW (`lr=1e-3`, `weight_decay=1e-4`), StepLR (step 30, gamma 0.5), and 50 epochs. Train fresh models for seeds `0`, `1`, and `2` on CPU only.

CPU is required because the v1 Apple MPS run produced NaN losses. No model selection, early stopping, architecture change, or hyperparameter change is permitted.

The reference runtime is Python 3.11.15, PyTorch 2.13.0, and
`neuraloperator==2.0.0`; the complete v1 environment record is retained in
`evidence/reproducibility.md`. A future executor must record any runtime
deviation in the P1 manifest before starting training.

## Metrics and decision rule

For each seed and each resolution, compute the mean relative L2 error over the 200 paired examples. Aggregate each metric as the arithmetic mean across the three seeds and report the sample standard deviation for 16x16. Define `gap = mean_relative_l2_res32 - mean_relative_l2_res16`.

P1 is a **success** only when all of the following hold:

1. mean 16x16 relative L2 is below `0.05`;
2. `gap <= 0.05`; and
3. 16x16 standard deviation is at most 30% of its mean.

P1 is a **failure** if mean 16x16 relative L2 exceeds `0.15` or `gap > 0.10`. All other results are **inconclusive**. This explicit gap rule is newly predeclared for P1 because v1's phrase “small margin” was not numeric; it does not retroactively alter v1.

## Independent reproduction and artifacts

Before execution, a reproducer must verify the source file SHA-256, source shape `(1000, 32, 32)` for both `x` and `y`, the `[200:400]` index slice, and the constructed 16x16 shape. An execution implementation must record those checks in `outputs/p1_paired_200/protocol_manifest.json` before training.

The outcome run is then exactly three fresh seed runs followed by aggregation. The implementation must emit:

```text
outputs/p1_paired_200/protocol_manifest.json
outputs/p1_paired_200/results_seed0.json
outputs/p1_paired_200/results_seed1.json
outputs/p1_paired_200/results_seed2.json
outputs/p1_paired_200/aggregate.json
outputs/p1_paired_200/sha256sums.txt
```

The final report must state that P1 measures a paired constructed-resolution effect. It must not describe P1 as native 16x16-versus-native 32x32 evidence.

## Freeze and amendment rule

The config and this protocol document must be committed before any outcome run. Any modification to the data source, file hash, index range, interpolation, architecture, training schedule, seeds, metric, or decision rule creates a new protocol ID and requires a new pre-execution commit and review. No such modification can overwrite or relabel v1 or the exploratory matched-count package.
