# Evaluation Protocol

This protocol is committed before model training. Any post-test change to data, hyperparameters,
calibration or thresholds creates a new experiment version and requires a new untouched evaluation
source.

## Research questions

1. Can LoRA-RoBERTa improve fine-grained banking intent routing over inexpensive baselines?
2. Does parameter-efficient tuning retain useful performance relative to full fine-tuning on this
   Mac-first workflow?
3. Does temperature scaling improve probabilistic calibration without changing class predictions?
4. Can selective routing reduce error on automated cases at an operationally useful coverage?
5. Do privacy and risk controls behave correctly independently of model accuracy?

## Data policy

- Preserve the official BANKING77 train and test boundary.
- Normalise Unicode, case and whitespace only for duplicate detection; do not rewrite model input.
- Quarantine official-training rows whose normalised text occurs in the official test file.
- Keep same-label duplicate groups together while deriving a group-stratified validation split
  only from official training data.
- Use validation data for checkpoint selection and temperature fitting.
- Use a separate development set for uncertainty and possible-OOD threshold selection.
- Keep official test and unknown-request holdout data untouched until all choices are locked.
- Record source revision, expected and observed file hashes, exact row indices, counts, label
  mappings and overlap checks in a text-free manifest.

## Comparisons

- TF-IDF plus logistic regression.
- Frozen RoBERTa embeddings plus a linear classifier.
- LoRA-adapted RoBERTa.
- Full RoBERTa fine-tuning if practical on the Mac.

The primary comparison uses seeds 17, 42 and 73 with an identical split policy.

The Module 3 seed-42 TF-IDF result is an implementation checkpoint. It must not be described as the
final primary comparison until the registered multi-seed module has been completed.

Module 4 follows the same rule. Validation-search amendments are permitted before test access only
when the reason, round and unchanged test-access state are recorded in the hashed configuration.
The final search round must be declared before test embeddings are created.

## Metrics

### Classification

- accuracy;
- macro-F1 and weighted-F1;
- per-intent precision, recall, F1 and support;
- worst-performing intents and confusion pairs;
- mean, standard deviation and seed-level results.

### Calibration

- negative log-likelihood;
- 15-bin expected calibration error;
- reliability diagrams before and after temperature scaling.

### Selective routing

- risk-coverage curve;
- known-request coverage at the locked threshold;
- false-automation rate;
- high-risk routing recall.

### Possible-OOD evaluation

- unknown-request recall;
- false acceptance rate;
- area under the in-vs-out detection curve where appropriate;
- separate reporting for synthetic, adversarial and external sources.

### Efficiency

- total and trainable parameters;
- wall-clock training time;
- process memory and available MPS memory measures;
- inference p50 and p95 latency on the documented Mac;
- adapter and full-checkpoint storage size.

## Robustness slices

- typographical corruption;
- paraphrases;
- multi-intent messages;
- code-switching;
- PII-containing messages;
- short and ambiguous messages;
- non-banking requests;
- configured fraud and security cases.

## Initial acceptance gates

| Area | Gate |
|---|---|
| Data integrity | No cross-split exact duplicates and no test use in selection |
| Reproducibility | All primary methods complete seeds 17, 42 and 73 |
| Calibration | Calibrated ECE is lower than raw ECE and no greater than 0.05 |
| Unknown handling | At least 90% recall on the labelled synthetic holdout |
| Policy safety | Every configured security-sensitive fixture goes to review or security |
| Privacy | No original or redacted message text in audit events |
| Evidence | Every public number maps to a versioned result artifact |

The gates are hypotheses and engineering targets, not achieved results.

## Claims policy

The repository may describe implemented controls before training. It may not claim improved
accuracy, production OOD detection, regulatory compliance or deployment readiness until evidence
for that exact claim exists and limitations are published beside it.
