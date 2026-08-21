# Model card: Governed Banking Intent Router

## Status and decision

This card describes the model inventory for the **v0.2.0 research preview**.

**No model is approved for production banking use.**

TF-IDF word-and-character logistic regression (`tfidf-word-char-c4`) is the registered champion.
The shadow API currently serves the revised rank-8 LoRA-RoBERTa challenger
(`lora-roberta-r8-revised`). That mismatch is intentional research instrumentation and a production
release blocker; it must not be described as champion deployment.

## Model inventory

| Model | Architecture | Registry role | Current disposition |
|---|---|---|---|
| TF-IDF word/character logistic regression | Sparse word and character n-grams with logistic regression | Champion | Retained from the historical like-for-like comparison |
| Frozen RoBERTa | Frozen `roberta-base` mean-pooled embeddings with logistic regression | Challenger | Historical; not promotion-eligible |
| Original LoRA-RoBERTa | `roberta-base` with rank-8 parameter-efficient adapters | Retired challenger | Underfit registered protocol |
| Revised LoRA-RoBERTa | Rank-8 adapters with multi-seed validation stopping | Active development challenger and shadow service model | Not promotion-eligible |
| Full RoBERTa | Full encoder fine-tuning | Planned CUDA challenger | Not evaluated |

The machine-readable decision is `configs/champion_challenger.yaml`; the generated registry is
`reports/champion/champion-registry.json`.

## Intended model task

The models assign one of 77 BANKING77 intent labels to a short, English banking-support message.
The classifier output is advisory metadata consumed by a deterministic policy. It is not an
authorization, fraud determination, identity decision, financial recommendation or transaction
instruction.

## Training and evaluation data

Training and development use the pinned public BANKING77 source described in `docs/DATA_CARD.md`.
The loader quarantines exact normalized train/test overlaps and preserves the official test split.
Synthetic fixtures test PII controls, routing, possible-OOD behaviour, service integration and
robustness. No real customer data is permitted in the research preview.

The official BANKING77 test set has already been observed. It cannot be used again as fresh
confirmation for tuned candidates. Promotion requires a new, locked external dataset before model
or threshold access.

## Registered results

| Evidence | Result | Interpretation |
|---|---:|---|
| TF-IDF historical official-test macro-F1 | 0.9053 | Current champion; single-seed observed benchmark evidence |
| Frozen RoBERTa historical official-test macro-F1 | 0.8964 | Did not outperform champion |
| Original LoRA historical official-test macro-F1 | 0.8202 | Registered protocol underfit |
| Revised LoRA three-seed validation macro-F1 | 0.8974 mean | Post-test development evidence, not independent confirmation |
| Revised LoRA scaled calibration-assessment ECE | 0.0280 mean | Improved selected measures on reused validation pools |
| Revised LoRA synthetic possible-OOD AUROC | 0.9611 mean | Ranking evidence only; operating thresholds failed |
| Synthetic robustness acceptable-intent accuracy | 68.52% | Failed the registered 80% gate |
| Synthetic robustness expected-security routing recall | 78.57% | Failed the registered 100% gate |

Uncertainty is diagnostic only. All seeds exceeded the registered 5% selective-risk ceiling; two
seeds also missed the synthetic possible-OOD recall target. The policy therefore cannot use these
scores to authorize a normal suggestion.

## Input and output

- Input: one bounded UTF-8 support message, transiently held in application memory.
- Preprocessing: structured-PII redaction before model inference.
- Model output: intent label and probability vector.
- Policy output in the current mode: `human_review` or `security_queue` only.
- Persisted evidence: allowlisted metadata, versions, categorical PII counts and routing outcome;
  never source text, redacted text, exact length or a message hash.

## Ethical and safety considerations

The benchmark does not establish regional, demographic, dialect, disability, multilingual or
customer-segment performance. A 77-class closed-set taxonomy can force unfamiliar or multi-intent
requests into an inappropriate label. Security misrouting can delay urgent assistance; false
security escalation can increase workload. Human review remains mandatory, and reviewers must be
able to disregard the prediction without penalty.

## Known limitations

- Public English benchmark data is not representative production traffic.
- The currently served model is not the champion.
- The robustness and uncertainty operating-point gates failed.
- Calibration assessment reused pools previously involved in checkpoint development.
- Regex PII detection misses names and contextual identifiers.
- No representative fairness analysis is possible from the available data.
- Real CUDA, Linux serving-container and identity-gateway operation remain unverified.
- Model explanations are not supplied and probabilities are not guarantees of correctness.

## Required human and technical controls

Use is permitted only under `docs/governance/INTENDED_USE.md` and
`docs/governance/HUMAN_OVERSIGHT.md`. Model changes require the evidence and approvals in
`docs/governance/VALIDATION_REVALIDATION_POLICY.md` and `docs/governance/CHANGE_APPROVAL.md`.
Operational risks and monitoring requirements are recorded in `governance/risk-register.yaml` and
`docs/governance/MONITORING_PLAN.md`.

The consolidated experimental protocol and results are maintained in `docs/EVALUATION.md`.

## Card maintenance

- Owner: `model_risk_reviewer`
- Accountable organisation: INNETWORK Technology Limited
- Review: every release, after a material model/data/policy change, after a relevant incident and
  at least every 90 days while active
- Version: `v0.2.0-research-preview`
- Effective date: 2026-08-21
