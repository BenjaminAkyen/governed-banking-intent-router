# Selective Prediction and Possible-OOD Card

## Decision

Module 8 tested whether calibrated uncertainty signals could support an abstention boundary for the
LoRA banking-intent classifier. The registered operating-point gates **did not pass**. The project
must not use these thresholds to approve automated routing.

## Claim boundary

This is post-calibration, post-test exploratory evidence. Known rows came from Module 7's
`calibration_assessment` role and had earlier been used for Module 6 checkpoint selection. Module 8
did not load BANKING77's official test split.

The possible-OOD fixture contains 96 authored non-banking messages across 12 domains. It contains
no customer data, but it is synthetic and cannot estimate the prevalence, language, ambiguity or
adversarial behaviour of real unknown requests. “Possible OOD” is a screening signal, not a factual
determination that a message is outside the training distribution.

## Role separation

Known rows were repartitioned by normalized-text group:

| Seed | Threshold development | Selective assessment | Group/index overlap |
|---:|---:|---:|---:|
| 17 | 375 | 375 | 0 / 0 |
| 42 | 375 | 376 | 0 / 0 |
| 73 | 375 | 375 | 0 / 0 |

Synthetic possible-OOD rows were partitioned by scenario group, stratified across all 12 domains:

| Role | Rows | Scenario groups | Cross-role group overlap |
|---|---:|---:|---:|
| `threshold_ood_development` | 48 | 24 | 0 |
| `possible_ood_assessment` | 48 | 24 | 0 |

The role registry was created before model scoring. Signal and threshold selection used development
roles only; every result below uses assessment roles only.

## Registered selection rule

Three calibrated signals competed independently for each seed:

- maximum predicted probability;
- top-two probability margin;
- one minus normalized predictive entropy.

Higher values mean “more like a confident known request.” A development threshold was feasible only
when it simultaneously achieved:

- at least 90% synthetic possible-OOD recall;
- no more than 5% selective risk among accepted known requests;
- at least 70% known-request coverage.

Among feasible candidates, the locked rule maximized known coverage, then possible-OOD recall, then
minimized selective risk. Registration order resolved any remaining signal tie. The coverage floor
prevents the misleading reject-everything solution.

## Locked assessment results

| Seed | Signal | Threshold | Coverage | Selective risk | Error capture | Possible-OOD recall | AUROC | Gate |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 17 | Inverse normalized entropy | 0.7378 | 0.9200 | 0.0609 | 0.4474 | 0.8542 | 0.9546 | Fail |
| 42 | Maximum probability | 0.5916 | 0.9362 | 0.0597 | 0.3636 | 0.8750 | 0.9586 | Fail |
| 73 | Inverse normalized entropy | 0.7666 | 0.8827 | 0.0725 | 0.4894 | 0.9375 | 0.9699 | Fail |
| **Mean** | — | — | **0.9129** | **0.0643** | **0.4335** | **0.8889** | **0.9611** | **Fail** |

All seeds exceeded the 70% coverage floor. Every seed exceeded the 5% selective-risk ceiling. Seeds
17 and 42 missed the 90% possible-OOD recall target; only seed 73 met it. The result must remain a
failed gate even though some bootstrap intervals include the target.

## Why AUROC did not make the threshold safe

Mean possible-OOD AUROC was 0.9611, showing useful ranking over this synthetic fixture. AUROC
averages performance across every possible threshold, however. A deployed router needs one locked
operating point. At the thresholds chosen on development data, ranking quality did not translate
into the registered combination of coverage, selective risk and possible-OOD recall.

This distinction is operationally important: a compelling discrimination chart can coexist with an
unsafe automation boundary.

## Failure analysis

Across the 48-message possible-OOD assessment role, public-services requests were the weakest
domain: recall was 0.25, 0.50 and 0.50 across seeds. Shopping/delivery requests reached only 0.50,
0.75 and 0.75. Their vocabulary resembles banking support language—“replacement,” “delivery,”
“address,” and “damaged”—and sometimes produced confident banking labels.

Assessment selective risk ranged from 5.97% to 7.25%. The locked rules captured only 36.4%–48.9%
of known-request errors, leaving too many confident mistakes among accepted requests.

The selected signal also changed by seed. Entropy won for seeds 17 and 73, while maximum probability
won for seed 42. That instability is another reason not to hard-code one of these thresholds into a
service.

## Supported and prohibited claims

Supported:

> On disjoint exploratory assessment roles, the calibrated scores ranked synthetic non-banking
> messages below known BANKING77 messages with mean AUROC 0.9611, but the locked operating points
> failed the registered selective-risk and possible-OOD gates.

Prohibited:

- the model detects real-world OOD traffic;
- high AUROC makes automated routing safe;
- the thresholds generalize to another bank, language or channel;
- rejected requests are necessarily OOD;
- accepted requests are correct or low risk.

## Engineering decision

Module 9 may consume uncertainty metadata, but it must default uncertain requests to human review
and must not treat the failed Module 8 threshold as production approved. Before deployment, the
project needs representative, independently collected known and unknown traffic, pre-registered
operating costs, stronger uncertainty methods and operational review-capacity analysis.

## Evidence

- Configuration: `configs/uncertainty.yaml`
- Synthetic fixture: `data/fixtures/synthetic-possible-ood.jsonl`
- Role registry: `data/manifests/banking77-uncertainty-index.json`
- Seed reports: `reports/uncertainty/seed-*-selective-ood.json`
- Aggregate: `reports/uncertainty/selective-ood-aggregate.json`
- Guided audit: `output/jupyter-notebook/08-selective-prediction-possible-ood.ipynb`
