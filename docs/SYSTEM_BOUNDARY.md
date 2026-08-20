# System Boundary

## Intended user and decision

The intended operator is a customer-support or security team reviewing inbound support requests.
The system may recommend one of three routing actions:

- `suggest_queue`: propose a normal support queue;
- `human_review`: defer an uncertain or policy-sensitive request;
- `security_queue`: prioritise a configured security-sensitive request.

The recommendation is advisory. The receiving organisation remains responsible for every customer
and account decision.

## In scope

- English-language research evaluation using BANKING77.
- Structured PII redaction before inference.
- Fine-grained intent prediction using RoBERTa and LoRA.
- Post-hoc temperature scaling fitted on validation logits.
- Confidence, top-two margin, entropy and possible-OOD signals.
- Versioned deterministic rules that combine intent risk and uncertainty.
- Metadata-only audit events with traceability to model and policy versions.
- Human correction capture as a separately reviewed feedback source.

## Out of scope

- Executing or approving a banking transaction.
- Freezing or unfreezing an account.
- Authenticating a customer or deciding account ownership.
- Giving financial, legal or regulatory advice.
- Replacing fraud, complaints, vulnerability or safeguarding teams.
- Training on real customer data during this research project.
- Claiming production readiness from BANKING77 performance.

## Data flow and minimisation

1. A message enters transient application memory.
2. Structured identifiers are detected and replaced before model inference.
3. The classifier receives only the redacted representation.
4. The policy consumes labels, scores, uncertainty signals and configured risk tiers.
5. The audit event stores decision metadata, hashes and version identifiers, not message text.
6. Any human correction enters a quarantined review process rather than automatic retraining.

## Failure consequences

| Failure | Potential consequence | Primary control |
|---|---|---|
| Fraud request sent to a normal queue | Delayed security response | Risk override and security queue |
| Unknown request forced into a known class | Incorrect handling | Possible-OOD signal and abstention |
| Overconfident wrong prediction | Unsafe automation | Calibration and risk-coverage policy |
| Personal data written to logs | Privacy incident | Metadata-only schema and privacy tests |
| Feedback automatically becomes training data | Poisoning or label drift | Quarantine and human approval |
| Model update changes routing behaviour | Unreviewed policy change | Separate model and policy versions |

## Human oversight requirements

A future deployment must define review ownership, response times, reviewer access, override reasons,
appeal paths and monitoring of reviewer workload. The phrase "human in the loop" is not treated as a
control unless those operational details are documented and tested.
