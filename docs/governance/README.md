# Governance and assurance

The governance set defines the current research boundary, accountable roles, release vetoes and
operational procedures. It is evidence of documented controls—not regulatory certification,
production approval or proof that the controls are operational in a bank.

## Current decision

| Attribute | Value |
|---|---|
| Release target | v0.2.0 research preview |
| Production approved | No |
| Real customer data permitted | No |
| Autonomous routing permitted | No |
| Current mode | `shadow_review_only` |
| Release approval | Pending |

The machine-readable contract is `configs/governance/module17.yaml`. The filename is retained as a
stable historical identifier; the maintained public documentation is organised by concern rather
than development module.

## Documents

| Document | Control objective |
|---|---|
| [Intended and prohibited use](INTENDED_USE.md) | Define the allowed research scope and stop conditions |
| [Risk register](RISK_REGISTER.md) | Explain scoring, ownership, treatments and release impact |
| [Threat model](governed-banking-intent-router-threat-model.md) | Record assets, trust boundaries, abuse paths and mitigations |
| [Human oversight](HUMAN_OVERSIGHT.md) | Preserve human authority, escalation and reviewer-capacity requirements |
| [Incident response](INCIDENT_RESPONSE.md) | Define detection, containment, evidence preservation and closure |
| [Rollback](ROLLBACK_PROCEDURE.md) | Restore an immutable prior release without hot mutation |
| [Monitoring](MONITORING_PLAN.md) | Define privacy-safe indicators, responses and review cadence |
| [Validation and revalidation](VALIDATION_REVALIDATION_POLICY.md) | Define evidence layers, release gates and revalidation triggers |
| [Change approval](CHANGE_APPROVAL.md) | Define change classes, required reviewers and decision records |
| [NIST AI RMF mapping](NIST_AI_RMF_MAPPING.md) | Map current evidence and gaps to Govern, Map, Measure and Manage |

The formal [system](../SYSTEM_CARD.md), [model](../MODEL_CARD.md) and
[data](../DATA_CARD.md) cards are mandatory release documents. The
[claims register](../CLAIMS_REGISTER.md) constrains public statements.

## Accountable roles

| Role | Authority |
|---|---|
| Accountable system owner | Own scope and resources; veto or stop release |
| Model-risk reviewer | Challenge model, data and evaluation evidence; veto promotion |
| Security and privacy reviewer | Challenge security/privacy controls; initiate incident response |
| Service operator | Disable or roll back an approved revision; cannot silently alter policy or model |
| Human queue reviewer | Override advisory output; cannot act on router output alone |

Named assignments and independent review are not yet recorded. An empty approval list is
intentional and must remain empty until real reviewers make attributable decisions.

## Maintenance

Review this set before every release, after material changes or incidents, and at least every 90
days while the project remains active. A lower risk score, broader intended use or removed release
blocker requires new evidence and a change record; prose changes alone cannot create approval.
