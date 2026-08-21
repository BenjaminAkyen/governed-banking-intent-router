# Module 12 Champion–Challenger Card

## Decision

TF-IDF word-and-character logistic regression remains the current champion. No challenger is
eligible for promotion, and no production deployment is approved.

This is a model-governance decision, not a claim that TF-IDF is universally better than RoBERTa.
It is the strongest model in the only completed like-for-like official-test comparison. That test
set is now observed and cannot be reused to confirm any subsequently tuned model.

## Current registry

| Model | Registry role | Evidence status | Promotion status |
|---|---|---|---|
| TF-IDF word+character logistic regression | Champion | Historical seed-42 official test | Retained |
| Frozen RoBERTa mean embeddings + logistic regression | Challenger | Historical seed-42 official test | Ineligible evidence |
| Original rank-8 LoRA-RoBERTa | Retired challenger | Historical seed-42 official test | Ineligible evidence |
| Revised rank-8 LoRA-RoBERTa | Active development challenger | Three-seed validation, calibration and synthetic possible-OOD | Ineligible evidence |
| Full RoBERTa fine-tuning | Planned CUDA challenger | No completed run | Not evaluated |

Historical BANKING77 test macro-F1 was 0.9053 for TF-IDF, 0.8964 for frozen RoBERTa and 0.8202 for
the original LoRA protocol. These results explain the registry state but are not a new Module 12
evaluation.

## Revised LoRA evidence

The revised LoRA protocol provides useful development evidence:

- mean validation macro-F1: 0.8974 across seeds 17, 42 and 73;
- mean calibrated assessment ECE: 0.0280;
- mean synthetic possible-OOD AUROC: 0.9611; and
- mean selective risk at the locked thresholds: 0.0643.

Calibration's registered point-estimate gates passed. The uncertainty operating-point gates did
not: every seed exceeded the 0.05 selective-risk ceiling, while seeds 17 and 42 also missed the
0.90 synthetic possible-OOD recall target. Moreover, the validation roles had already influenced
checkpoint selection. These results cannot promote LoRA or be directly compared with TF-IDF's
official-test macro-F1.

## Promotion routes

Every route requires seeds 17, 42 and 73 on the same new locked external dataset.

### Route A: classification superiority

- positive challenger-minus-champion macro-F1 for every seed; and
- a 95% paired-bootstrap confidence-interval lower bound above zero for the mean difference.

### Route B: non-inferior classification plus operational value

- mean macro-F1 difference confidence-interval lower bound at least -0.005;
- no seed more than 0.010 below the champion; and
- either:
  - mean ECE reduction of at least 0.010, no seed worse and paired uncertainty supporting lower
    ECE; or
  - mean selective-risk reduction of at least 0.020 at matched coverage, with every seed reaching
    0.70 known coverage and 0.90 representative possible-OOD recall, plus paired uncertainty
    supporting lower risk.

Both routes are vetoed if the minimum security-intent F1 difference is below -0.020 or if privacy,
routing or audit controls fail. Passing makes a challenger eligible for human review; the software
cannot promote it automatically.

## Data required for a credible decision

The new evaluation source must be external to BANKING77 development and test material. Before any
candidate sees its text or labels, create a versioned lock containing:

- ownership, provenance, licence and collection period;
- immutable row identifiers, label taxonomy, row count and SHA-256;
- duplicate and near-duplicate checks against BANKING77;
- class, language, ambiguity and security-risk coverage;
- a lawful PII-handling basis and access controls;
- preprocessing, calibration, uncertainty and routing-policy hashes; and
- a timestamped declaration that model and threshold development is complete.

Synthetic cases remain useful for control testing, but they cannot replace a representative
external comparison.

## Service alignment

Module 10 serves the revised seed-42 LoRA adapter in `shadow_review_only` mode. The served model is
not the registry champion. This is acceptable for a bounded research shadow service, but it is a
production release blocker. A later deployable service must load the approved champion through a
versioned registry reference, retain rollback support and refuse startup when its model hash does
not match the approved decision.

## Reproduction

Build and validate the metadata-only registry without loading dataset text:

```bash
python scripts/build_champion_registry.py
pytest -q tests/test_champion.py
```

The canonical files are:

- `configs/champion_challenger.yaml`;
- `reports/champion/champion-registry.json`;
- `src/governed_banking/champion.py`;
- `docs/decisions/0003-champion-challenger-promotion.md`; and
- `output/jupyter-notebook/12-champion-challenger-registry-audit.ipynb`.

## Current limitation and next step

Module 12 has established the decision machinery, but it has not created new challenger model
results. The next engineering step is to register and implement validation-only full RoBERTa
fine-tuning on real CUDA and comparable three-seed development runs for TF-IDF and frozen RoBERTa.
Those development comparisons may guide which models proceed, but final promotion still requires
the separately locked external evaluation source.
