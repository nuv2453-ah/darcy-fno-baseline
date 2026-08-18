# Project Brief: FNO Baseline for 2D Darcy Flow

**Track:** Physics-Informed AI & Neural Operators
**Lead:** Nuv Ahuja
**Status:** Frozen original v1 completed; post-hoc matched-count control retained

## Research Question

Can a Fourier Neural Operator (FNO), trained on a fixed number of 2D Darcy
flow samples at one resolution, learn the coefficient-to-solution operator
accurately enough to generalize *zero-shot* to a higher spatial resolution
it never saw during training?

## Hypothesis (falsifiable)

An FNO trained on N=1000 samples at 16×16 resolution achieves relative L2
error below 5% on a held-out 16×16 test set, **and** the error at 32×32
(unseen resolution) does not increase by more than a small, pre-registered
margin. If in-distribution error exceeds 15%, or the model fails to transfer
to 32×32 at all (error roughly doubles or worse), the hypothesis is
rejected for this architecture/data regime.

## Literature (5–8 papers)

1. **Li et al., "Fourier Neural Operator for Parametric PDEs" (2020).**
   Introduces FNO; learns a resolution-invariant operator by parameterizing
   convolutions in Fourier space, evaluated on Darcy flow and
   Navier-Stokes. This baseline reproduces their Darcy flow setup directly.
2. **Lu et al., "DeepONet: Learning nonlinear operators..." (2021).**
   Alternative operator-learning architecture (branch/trunk network); a
   natural second baseline once the FNO pass is done.
3. **Kovachki et al., "Neural Operator: Learning Maps Between Function
   Spaces" (2021/2023).** Theoretical framing of neural operators as
   discretization-invariant maps — motivates why resolution generalization
   is a meaningful test rather than an incidental property.
4. **Takamoto et al., "PDEBench: An Extensive Benchmark for Scientific
   Machine Learning" (2022).** Standardized PDE datasets and evaluation
   protocol; used here as a fallback/cross-check dataset source.
5. **Gupta & Brandstetter, "Towards Multi-spatiotemporal-scale Generalized
   PDE Modeling" / PDEArena (2022).** Benchmark and discussion of
   generalization across resolutions and PDE families — relevant to the
   OOD test design.
6. **Li et al., "Fourier Neural Operator with Learned Deformations for PDEs
   on General Geometries" / geo-FNO (2022).** Notes limitations of vanilla
   FNO (regular grids) — relevant if a later iteration moves to irregular
   domains.
7. **Tran et al., "Factorized Fourier Neural Operators" (2023).** Efficiency
   and regularization improvements to FNO — candidate next step if the
   baseline underfits or overfits.

## Baseline Model

Fourier Neural Operator (FNO), via the official `neuraloperator` library
(same implementation as the original paper's authors). Configuration:
4 Fourier layers, 32 hidden channels, 16 Fourier modes per dimension.

## Dataset & Protocol

- **Source:** Small Darcy flow dataset bundled with `neuraloperator`
  (Zenodo-hosted, same data as the original FNO paper).
- **Declared split:** 1000 train / 200 test @ 16×16 (in-distribution) /
  200 test @ 32×32 (resolution-generalization, zero-shot).
- **Observed frozen v1 split:** 1000 train / 50 test @ 16×16 /
  200 test @ 32×32. The 16×16 source test file contains only 50 examples;
  the original v1 result and its hashes are preserved unchanged.
- **Seeds:** 3 runs (seeds 0, 1, 2) to report variance, not a single
  cherry-picked run.
- Full details are pinned in `configs/default.yaml` and fixed *before*
  the main run.

## Protocol history

- **Frozen original v1:** trained and evaluated with the observed 50 samples
  at 16×16 and 200 samples at 32×32; verdict **INCONCLUSIVE**.
- **Post-hoc exploratory matched-count control**
  (`matched_test_n50_successor_v1`): evaluated the retained checkpoints on
  50 samples at each resolution, with no retraining. It was run after the
  original 50-vs-200 outcome was known, so it is not pre-registered or
  confirmatory evidence. Its JSON and hash are retained unchanged.
- **Future confirmatory work:** any 200-vs-200 evaluation or revised
  cross-resolution transfer criterion must be separately versioned, with the
  hypothesis, metrics, data cardinalities/provenance, and decision rule
  frozen before execution.

## Metrics (fixed before the main run)

- **Primary:** Relative L2 error on the 16×16 (in-distribution) test set.
- **Secondary:** Resolution-generalization gap = error@32×32 − error@16×16.

## Compute Budget

FNO at this scale trains in well under an hour per seed on a single
consumer/cloud GPU (or a free-tier Colab/Kaggle GPU). Total budget for
3 seeds plus reruns: ~5–10 GPU-hours.

## Success / Failure / Inconclusive

- **Success:** In-distribution relative L2 < 5%, and the resolution gap is
  small (model doesn't collapse at 32×32).
- **Failure:** In-distribution relative L2 > 15%, or the model fails to
  transfer to 32×32 at all.
- **Inconclusive:** Result falls between the two thresholds, or variance
  across seeds is too high (std > 30% of the mean) to draw a conclusion.

## Repo

Code-complete baseline (data loading, model, train, eval, pre-registered
thresholds) is in this repo. Training run itself requires normal internet
access to pull the Zenodo-hosted dataset.
