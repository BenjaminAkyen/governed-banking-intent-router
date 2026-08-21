# Evaluation

| Attribute | Value |
|---|---|
| Status | Completed research evidence with unresolved promotion gates |
| Primary benchmark | BANKING77 |
| Current champion | TF-IDF word + character logistic regression |
| Active challenger | Revised rank-8 LoRA-RoBERTa |
| Independent confirmation set | Not available |

## Evaluation principles

The evaluation design separates model development, probability calibration, threshold selection
and assessment. The official BANKING77 test split has already been observed and is historical
evidence. It must not be reused to confirm models tuned afterwards.

The project follows these rules:

- preserve the official train/test boundary;
- group exact normalized duplicates before creating validation roles;
- fit features, models, temperatures and thresholds only on their registered development roles;
- record source, configuration, code and artifact hashes;
- keep reports free of message text;
- publish failures and uncertainty alongside point estimates; and
- require a new locked external dataset for any fresh promotion claim.

## Data roles

The BANKING77 loader verifies the pinned source, quarantines seven normalized train/test overlaps
from training and retains all 3,080 official test rows. The seed-42 development split contains
8,495 training rows and 1,501 validation rows with no final normalized-text overlap.

Later studies use additional group-safe roles:

| Purpose | Development role | Assessment role | Limitation |
|---|---|---|---|
| Model selection | Training data | Validation data | Validation influences checkpoint choice |
| Temperature scaling | `temperature_fit` | `calibration_assessment` | Both originate from a previously used validation pool |
| Uncertainty threshold | `threshold_known_development` and synthetic OOD development | `selective_known_assessment` and synthetic OOD assessment | Synthetic OOD is not representative traffic |
| Robustness | Frozen synthetic pack v1 | Same pack, first observation | Pack is now observed and cannot become a fresh confirmation set |

See the [data card](DATA_CARD.md) for provenance and fixture restrictions.

## Model comparison

All three historical models used the same 3,080 official test rows. These are single-seed results.

| Model | Accuracy | Macro-F1 | Log loss | Top-3 accuracy |
|---|---:|---:|---:|---:|
| TF-IDF word + character logistic regression | **0.9052** | **0.9053** | 0.4970 | 0.9705 |
| Frozen RoBERTa mean embeddings + logistic regression | 0.8961 | 0.8964 | **0.3990** | **0.9718** |
| Original rank-8 LoRA-RoBERTa | 0.8273 | 0.8202 | 0.7746 | 0.9497 |

TF-IDF remains champion because it has the strongest completed like-for-like classification result.
Frozen RoBERTa corrected 135 rows missed by TF-IDF, while TF-IDF corrected 163 rows missed by the
frozen model. The two-sided exact McNemar p-value was 0.1177; that does not establish equivalence.

The original LoRA protocol trained 944,717 parameters (0.7519% of the model) and was still improving
at its three-epoch boundary. The defensible conclusion is that the registered protocol underfit,
not that LoRA is generally inferior.

Evidence:

- `reports/baseline/tfidf-logreg-test.json`
- `reports/frozen-roberta/test.json`
- `reports/frozen-roberta/paired-vs-tfidf.json`
- `reports/lora-roberta/test.json`

## Revised LoRA development study

The revised protocol used seeds 17, 42 and 73 and validation-only stopping. It did not run another
official-test evaluation.

| Seed | Best epoch | Validation macro-F1 |
|---:|---:|---:|
| 17 | 8 | 0.9006 |
| 42 | 8 | 0.9026 |
| 73 | 6 | 0.8890 |
| **Mean ± sample SD** | — | **0.8974 ± 0.0073** |

Seeds 17 and 42 peaked at the eight-epoch boundary, so convergence is not established. These values
cannot be compared directly with the historical test results as evidence of improved
generalisation.

Evidence: `reports/multiseed-lora/` and `data/manifests/banking77-multiseed-index.json`.

## Calibration

One positive scalar temperature was fitted for each seed. Assessment used disjoint
`calibration_assessment` rows; positive scaling did not change any predicted labels.

| Metric | Raw mean | Scaled mean | Interpretation |
|---|---:|---:|---|
| Expected calibration error | 0.0470 | **0.0280** | Lower point estimate |
| Negative log-likelihood | 0.3507 | **0.3390** | Lower for every seed |
| Multiclass Brier score | 0.1456 | **0.1427** | Lower for every seed |

Paired bootstrap intervals supported lower NLL and Brier score for every seed. The seed-73 interval
for ECE change crossed zero, and maximum calibration error was unstable. The result supports a
narrow claim about selected assessment metrics, not production calibration.

Evidence: `reports/calibration/temperature-scaling-aggregate.json`.

## Selective prediction and possible OOD

Maximum probability, top-two margin and inverse normalized entropy competed on development roles.
The selected operating point had to achieve at least 90% synthetic possible-OOD recall, no more
than 5% selective risk and at least 70% known-request coverage.

| Seed | Selected signal | Coverage | Selective risk | Possible-OOD recall | AUROC | Gate |
|---:|---|---:|---:|---:|---:|---|
| 17 | Inverse normalized entropy | 0.9200 | 0.0609 | 0.8542 | 0.9546 | Fail |
| 42 | Maximum probability | 0.9362 | 0.0597 | 0.8750 | 0.9586 | Fail |
| 73 | Inverse normalized entropy | 0.8827 | 0.0725 | 0.9375 | 0.9699 | Fail |
| **Mean** | — | **0.9129** | **0.0643** | **0.8889** | **0.9611** | **Fail** |

High AUROC did not produce a safe locked threshold. Every seed exceeded the selective-risk ceiling,
and two missed the possible-OOD recall target. Signal choice also changed by seed. The policy may
record these values only as experimental review metadata.

Evidence: `reports/uncertainty/selective-ood-aggregate.json`.

## Synthetic robustness

The versioned pack contains 60 project-authored cases across typo, speech-transcription,
paraphrase, multi-intent, ambiguity, code-switching, PII, prompt-like manipulation, non-banking and
high-risk security families. It contains no customer data. A registered lexical check found no
exact or character-five-gram near matches within the pack or across 13,083 pinned BANKING77 rows.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| In-scope acceptable-intent rate | ≥ 80% | 68.52% (37/54) | Fail |
| Expected-security routing recall | 100% | 78.57% (11/14) | Fail |
| Overall routing-action agreement | ≥ 95% | 90.00% (54/60) | Fail |
| PII expectation agreement | 100% | 100% (60/60) | Pass |
| Suggestion actions | 0 | 0 | Pass |
| Message values or hashes in report | 0 | 0 | Pass |

Typographical errors were the weakest family at 16.67% acceptable-intent accuracy. Three expected
security cases were under-routed; three unrelated messages were over-routed to security after the
closed-set model forced a banking label. These are failure-discovery results from six cases per
family, not population estimates.

Evidence: `data/robustness/v1/` and `reports/robustness/module13-lora-mps-assessment.json`.

## Champion–challenger decision

No challenger is eligible for promotion. Passing numeric gates would create eligibility for human
review; software must never update the champion automatically.

A credible promotion requires the champion and challenger to be frozen and evaluated across seeds
17, 42 and 73 on the same new external dataset. The dataset must be locked before model access and
record provenance, licence, collection period, hashes, duplicate checks, language and risk
coverage, and an approved privacy basis.

Two promotion routes are registered in `configs/champion_challenger.yaml`:

- **Classification superiority:** positive challenger-minus-champion macro-F1 for every seed and a
  95% paired-bootstrap lower bound above zero.
- **Non-inferior classification with operational value:** classification within the registered
  non-inferiority bounds plus material calibration or matched-coverage selective-risk improvement.

Both routes are vetoed by unacceptable security-intent regression or failed privacy, routing or
audit controls. The shadow API's LoRA model is intentionally not champion-aligned and therefore
cannot be presented as a deployable champion service.

## Reproduction map

| Evidence | Command |
|---|---|
| Data integrity | `python scripts/prepare_banking77.py` |
| TF-IDF baseline | `python scripts/run_tfidf_baseline.py` |
| Frozen RoBERTa | `python scripts/run_frozen_roberta_baseline.py` |
| LoRA baseline | `python scripts/run_lora_roberta.py` |
| Multi-seed LoRA | `python scripts/run_multiseed_lora.py` |
| Calibration | `python scripts/run_temperature_scaling.py` |
| Uncertainty | `python scripts/run_uncertainty_evaluation.py` |
| Robustness | `python scripts/run_robustness_evaluation.py` |
| Champion registry | `python scripts/build_champion_registry.py` |

Model-backed commands require the registered local datasets and unredistributed model artifacts.
Some runs require explicit MPS or CUDA and fail if the requested backend is unavailable.

## Claim boundary

The evidence supports reproducible benchmark comparisons and implementation-level safety-control
tests. It does not establish representative banking performance, fairness, multilingual quality,
real-world OOD detection, production privacy recall, operational capacity, regulatory compliance or
fitness for autonomous routing. Consult the [claims register](CLAIMS_REGISTER.md) before publishing
results.
