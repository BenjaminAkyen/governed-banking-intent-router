# Risk register

The canonical, machine-readable register is `governance/risk-register.yaml`. This document explains
how it must be interpreted and operated.

## Scoring and tolerance

Likelihood and impact are scored from 1 to 5; the risk score is their product. Scores 1–4 are low,
5–9 medium, 10–16 high and 17–25 critical. The inherent score assumes no project controls. The
residual score reflects implemented controls, not planned mitigations.

For the research preview:

- no critical or high residual risk is accepted for production use;
- `open_release_blocker` prevents production approval but does not prevent labelled local research;
- `controlled_monitor` means controls exist but the risk remains active and must be monitored; and
- risk acceptance requires a change record naming the accountable role, rationale, expiry and
  compensating controls. The current register contains no accepted risks.

## Current risk summary

| ID | Risk | Residual | Treatment decision |
|---|---|---:|---|
| R-001 | Benchmark data is not representative | Critical 20 | Avoid production use; obtain governed external evaluation |
| R-002 | Uncertainty operating thresholds are unsafe | High 12 | Human review only; revalidate externally |
| R-003 | Served LoRA model is not the champion | High 12 | Research shadow use only |
| R-004 | Security-sensitive request is misrouted | High 15 | Mandatory human triage and external security-case validation |
| R-005 | PII or credentials leak | High 10 | Add production DLP and adversarial privacy evaluation |
| R-006 | Ambiguous or adversarial text is misrouted | High 12 | Expand multilingual and adversarial evaluation |
| R-007 | Model, policy or configuration is tampered with | High 10 | Sign releases and enforce reviewed protected branches |
| R-008 | Gateway misconfiguration exposes the origin | High 15 | Configure and test a real identity gateway |
| R-009 | Flooding exhausts serving capacity | Medium 9 | Add distributed limits and load evidence |
| R-010 | Audit or telemetry failure creates a blind spot | Medium 8 | Deploy durable stores and failure alerts |
| R-011 | Distribution change occurs without revalidation | High 12 | Establish representative baselines and reviewed thresholds |
| R-012 | Reviewers over-rely on advisory output | High 12 | Train reviewers and measure override/workload behaviour |
| R-013 | CI or dependency compromise alters a release | High 10 | Lock dependencies and sign/attest release artifacts |

## Review procedure

The accountable system owner reviews the register with model-risk and security/privacy reviewers:

- before every release;
- at least every 90 days;
- after a material model, data, policy, dependency, deployment or intended-use change;
- after a relevant incident, near miss, failed gate or new vulnerability; and
- before closing or accepting any risk.

Reviewers must update evidence paths, separate implemented controls from planned treatments and
record score changes in a change-approval record. A lower score without new evidence is prohibited.

## Release decision

The current register permits continued local research and open-source review. It does not permit a
production pilot, real customer data, direct internet exposure or autonomous routing.
