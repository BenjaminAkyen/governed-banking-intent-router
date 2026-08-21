# Temperature-Scaling Calibration Card

## Decision and claim boundary

Module 7 asks a narrow question: after a Module 6 LoRA checkpoint has been selected, can one
positive scalar temperature improve the quality of its reported probabilities on rows that did not
fit that temperature?

This is **post-selection, post-test exploratory evidence**. BANKING77's official test outcome was
observed in Module 5, and each source validation pool was previously used to select its Module 6
checkpoint. Module 7 does not load the official test split and does not provide an independent
estimate of model generalisation or production calibration.

## Leakage control

Each seed's validation pool is divided before calibration:

| Seed | `temperature_fit` | `calibration_assessment` | Index/group overlap |
|---:|---:|---:|---:|
| 17 | 749 | 750 | 0 / 0 |
| 42 | 750 | 751 | 0 / 0 |
| 73 | 749 | 750 | 0 / 0 |

The partition is label-stratified and deterministic. Rows with identical text after Unicode, case
and whitespace normalization remain in the same role. The registry stores source indices and
hashes, not customer-message text.

This design prevents the direct error of evaluating temperature scaling on the same probabilities
used to optimize temperature. It does not make the assessment independent of checkpoint selection,
because both roles originated in the earlier validation pool.

## Method

- Load each hash-verified Module 6 rank-8 LoRA checkpoint.
- Extract logits from validation rows on Apple MPS.
- Fit one positive scalar temperature on `temperature_fit` by minimizing multiclass negative
  log-likelihood over bounded log-temperature, with `0.05 ≤ T ≤ 10`.
- Apply that frozen temperature to `calibration_assessment`.
- Report negative log-likelihood (NLL), 15-bin fixed-width expected calibration error (ECE), maximum
  calibration error (MCE), multiclass Brier score, signed confidence gap and reliability bins.
- Compute 2,000 paired bootstrap resamples per seed for calibrated-minus-raw metric changes.

The fit role records only the optimizer's NLL objective. Published calibration metrics and
reliability bins are calculated exclusively on `calibration_assessment`.

Positive temperature scaling cannot change the ordering of logits. Prediction invariance is checked
explicitly rather than assumed.

## Assessment results

Lower values are better for ECE, NLL and Brier score.

| Seed | Temperature | ECE raw → scaled | NLL raw → scaled | Brier raw → scaled | Changed labels |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.8285 | 0.0555 → 0.0334 | 0.3378 → 0.3234 | 0.1354 → 0.1317 | 0 |
| 42 | 0.8910 | 0.0463 → 0.0265 | 0.3216 → 0.3090 | 0.1404 → 0.1375 | 0 |
| 73 | 0.8924 | 0.0391 → 0.0242 | 0.3926 → 0.3846 | 0.1611 → 0.1589 | 0 |
| **Mean** | **0.8706** | **0.0470 → 0.0280** | **0.3507 → 0.3390** | **0.1456 → 0.1427** | **0** |

Every seed passed the registered point-estimate gate: scaled ECE was both below raw ECE and no
greater than 0.05, with no changed class predictions. Temperatures below 1 sharpened probabilities;
the unscaled checkpoints were underconfident on average on these assessment partitions. Mean signed
confidence gap moved from -0.0365 to -0.0086.

## Uncertainty and counter-evidence

The paired bootstrap intervals for NLL and Brier-score change were below zero for all three seeds.
ECE-change intervals were below zero for seeds 17 and 42, while seed 73's interval crossed zero
(`[-0.0258, 0.0059]`). Accordingly, the point gate passed for seed 73 but its ECE improvement is not
robust to this resampling analysis.

MCE should not be used as the headline result. Its seed-level intervals were wide, and seed 17's
point estimate increased from 0.3080 to 0.3244. Sparse fixed-width bins can make maximum-error
estimates volatile.

## What this module supports

Supported statement:

> On disjoint calibration-assessment partitions drawn from the Module 6 validation pools, scalar
> temperature scaling improved mean ECE, NLL and multiclass Brier score across three seeds without
> changing predicted intent labels.

Unsupported statements:

- the classifier is calibrated for real bank traffic;
- temperature scaling improved official-test or independent generalisation;
- every calibration metric improved for every seed;
- a confidence threshold is now safe for automated routing;
- confidence is an out-of-distribution detector.

## Operational implication

Calibration changes confidence, not correctness. It is a prerequisite for evaluating selective
routing thresholds, not permission to automate customer requests. Module 8 must fit any uncertainty
or possible-OOD thresholds on a new development role and assess them on separate, representative
known and unknown-request data.

## Evidence

- Registered configuration: `configs/calibration.yaml`
- Partition registry: `data/manifests/banking77-calibration-index.json`
- Seed reports: `reports/calibration/seed-*-temperature-scaling.json`
- Aggregate: `reports/calibration/temperature-scaling-aggregate.json`
- Guided audit: `output/jupyter-notebook/07-temperature-scaling-calibration.ipynb`
