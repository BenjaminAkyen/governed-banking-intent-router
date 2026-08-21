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
- For Module 7, divide each Module 6 validation pool into a group-safe `temperature_fit` role and a
  disjoint `calibration_assessment` role before fitting temperature. Never assess calibration on
  the probabilities used to fit the temperature.
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

Module 5 completed the seed-42 official-test evaluation, so that benchmark outcome is now observed.
Module 6 is therefore post-test exploratory: it may create seeds 17, 42 and 73, revise stopping using
validation trajectories and aggregate validation results, but it must not compute another official-
test score or describe validation changes as improved generalisation. A new confirmatory performance
claim requires an independently sourced, untouched evaluation set and a frozen protocol.

Module 7 remains post-selection and post-test exploratory. For each seed, it partitions Module 6's
validation rows 50:50 by normalized-text group, fits one positive scalar temperature only on the
`temperature_fit` role, and reports calibration only on `calibration_assessment`. This separation
prevents temperature-fit resubstitution, but it does not undo the validation pool's earlier use for
checkpoint selection. Module 7 therefore cannot establish independent generalisation or production
calibration.

Module 8 partitions Module 7's `calibration_assessment` role again by normalized-text group into
`threshold_known_development` and `selective_known_assessment`. It separately partitions a
synthetic, non-customer possible-OOD fixture by scenario group into `threshold_ood_development` and
`possible_ood_assessment`. Signal choice and thresholds may use only the two development roles.
Risk-coverage, operating-point metrics and acceptance gates may use only the two assessment roles.
This prevents direct threshold resubstitution but does not erase the known rows' earlier selection
and calibration-assessment history.

Module 9 evaluates controls independently of model inference using versioned, authored synthetic
fixtures. Exact redacted strings and type counts are checked during execution, but neither source
nor redacted strings enter the summary report or audit sink. Routing cases assert expected action,
queue and precedence. The policy is hash-bound to Module 8's failed aggregate and treats its
thresholds as review-only observations. An allowlisted audit schema records coarse input-size
buckets, PII type counts, model and policy hashes, decisions and reason codes; it prohibits message
text, redacted text, message hashes, exact lengths and free-form fields.

Module 10 evaluates the local FastAPI integration with the hash-bound seed-42 adapter on MPS. A
versioned synthetic fixture covers known, ambiguous, possible-OOD, multi-intent, typographical,
code-switching, security-language and PII-bearing requests. Three warm-up calls precede three
measured repetitions of every fixture case. The registered local engineering targets are startup
at or below 30 seconds and p95 in-process API latency at or below 750 milliseconds.

The service evaluation also checks bearer authentication, trusted hosts, request-size limits,
schema rejection, disabled documentation, security headers, audit persistence and the absence of
source and redacted message values in API responses and audit events. These thresholds describe
one documented Mac and synthetic traffic; they are not service-level objectives or production
capacity evidence.

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
- 15-bin maximum calibration error;
- multiclass Brier score and signed confidence gap;
- reliability diagrams before and after temperature scaling.

### Selective routing

- risk-coverage curve;
- area under the risk-coverage curve;
- known-request coverage at the locked threshold;
- selective risk;
- false-automation rate;
- error-capture rate among abstained known requests;
- high-risk routing recall.

### Possible-OOD evaluation

- unknown-request recall;
- false acceptance rate;
- AUROC and average precision for known-vs-possible-OOD ranking;
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

Module 9 additionally requires all 11 registered detector classes to be exercised, every synthetic
expected redaction and route to match, zero `suggest_queue` actions, a validated audit round trip
and local evidence-sink permissions of `0700` for the directory and `0600` for the file. These are
control-verification gates, not estimates of real-world detector or policy effectiveness.

The gates are hypotheses and engineering targets, not achieved results.

## Claims policy

The repository may describe implemented controls before training. It may not claim improved
accuracy, production OOD detection, regulatory compliance or deployment readiness until evidence
for that exact claim exists and limitations are published beside it.
