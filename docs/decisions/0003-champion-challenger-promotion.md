# ADR 0003: Evidence-gated champion–challenger promotion

- Status: accepted
- Date: 2026-08-21
- Decision owner: project maintainer
- Applies from: Module 12

## Context

The official BANKING77 test split has already been observed. Its single-seed macro-F1 results are
0.9053 for TF-IDF logistic regression, 0.8964 for frozen RoBERTa and 0.8202 for the original LoRA
protocol. Those historical results support retaining TF-IDF as the current champion among the
models compared, but they cannot be reused as fresh evidence after further model development.

The revised three-seed LoRA work used BANKING77 validation roles after the official test result was
known. Its 0.8974 mean validation macro-F1 and later calibration results are useful development
evidence, not an independent promotion decision. Its registered uncertainty gates also failed.

Module 10 currently serves the seed-42 LoRA adapter in a local, review-only shadow service. That
implementation is not evidence that LoRA is the champion. The service must remain non-production
until its model source is aligned with an approved registry decision.

## Decision

1. Register `tfidf-word-char-c4` as the active historical champion.
2. Register frozen RoBERTa, revised LoRA-RoBERTa and future full-fine-tuned RoBERTa as challengers.
3. Treat historical official-test results as context only. Never use them for a new promotion.
4. Treat existing validation, calibration and synthetic possible-OOD results as development
   evidence only. They may identify promising challengers but cannot authorize promotion.
5. Require a new, versioned and locked external evaluation dataset before any challenger can be
   promoted. It must have documented provenance and licence, contain no customer data without an
   approved governance process, and remain unavailable to model, calibration and threshold
   development until every candidate is frozen.
6. Evaluate the champion and each challenger on the same locked rows, label taxonomy, preprocessing
   policy, calibration policy, routing policy and three seeds: 17, 42 and 73.
7. Record prediction-level, index-addressed results so classification, calibration, selective-risk
   and safety differences can be paired rather than compared as unrelated point estimates.
8. Keep the decision fail closed. Missing evidence, mismatched hashes, incomplete seeds, a reused
   development dataset or a failed safety veto produces `retain_champion`.

## Registered promotion routes

A challenger may pass one of two performance routes, but must also pass every common safety and
evidence gate.

### Route A: classification superiority

- challenger macro-F1 is higher than the champion for all three seeds; and
- the lower bound of the registered 95% paired-bootstrap confidence interval for the mean
  macro-F1 difference is greater than zero.

### Route B: non-inferior classification with operational improvement

- the lower bound of the registered 95% paired-bootstrap confidence interval for the mean
  macro-F1 difference is at least -0.005;
- no seed's macro-F1 is more than 0.010 below its champion counterpart; and
- either:
  - expected calibration error improves by at least 0.010 on average, is not worse for any seed and
    its paired confidence interval supports improvement; or
  - selective risk improves by at least 0.020 at matched coverage, every seed reaches at least 0.70
    known-request coverage and 0.90 representative possible-OOD recall, and the paired confidence
    interval supports lower risk.

### Common gates and vetoes

- all three seed runs and all required metrics are present;
- every artifact and dataset lock hash is valid;
- the external evaluation set has never been used for training, model selection, calibration or
  threshold selection;
- the official BANKING77 test split is not loaded or reported;
- worst registered security-intent F1 is no more than 0.020 below the champion;
- privacy, routing and audit safety tests pass;
- no result is silently pooled across incompatible datasets or preprocessing policies; and
- promotion is a recorded human approval, not an automatic registry mutation.

The numeric margins and statistical method are fixed before the external evaluation data exists.
They cannot be relaxed after a challenger result is observed.

## External evaluation lock

The future lock must record at minimum:

- dataset identifier, version, owner, provenance, licence and collection dates;
- immutable row IDs, row count, label taxonomy and dataset SHA-256;
- duplicate and near-duplicate analysis against all BANKING77 development and test material;
- class and risk-tier coverage;
- PII handling and approval basis;
- preprocessing and redaction configuration hashes;
- model, calibration, uncertainty and routing-policy hashes;
- the timestamp at which all candidates and gates were frozen; and
- an attestation that labels and texts were not accessed during development.

Until this lock exists, the only valid Module 12 decision is `retain_champion`.

## Consequences

- TF-IDF remains champion even though a transformer is more fashionable.
- The revised LoRA calibration improvement cannot offset failed uncertainty gates or the absence of
  comparable champion calibration evidence.
- Full RoBERTa fine-tuning may be developed on real CUDA hardware, but it remains a challenger and
  cannot be promoted using BANKING77 test data.
- Module 10's LoRA shadow service is explicitly model-registry misaligned and cannot be represented
  as the production champion deployment.
- Acquiring and governing an appropriate external evaluation set becomes a release dependency, not
  an optional follow-up.
