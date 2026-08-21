# Change-approval process and record

## Policy

Every release-affecting change requires a versioned record under `governance/change-approvals/`.
Approval is evidence of a bounded decision; it is not proof of regulatory compliance or safety.

## Change classes

| Class | Examples | Required review |
|---|---|---|
| Routine | Documentation correction with no claim, control or behaviour change | Change author plus repository checks |
| Control | Tests, audit, telemetry, authentication, dependency or deployment changes | Security/privacy reviewer and service operator |
| Model-risk | Model, data, split, label, preprocessing, calibration, uncertainty, threshold or policy changes | Independent model-risk reviewer and accountable system owner |
| Scope/high-risk | Real data, public exposure, multi-tenancy, autonomous routing, production pilot or customer-impacting use | All veto roles plus separate legal/privacy/security assessment |
| Emergency | Containment or rollback for an active incident | Incident commander may stop/roll back; retrospective approval required |

The author cannot be the sole approver for control, model-risk or scope/high-risk changes.

## Required record fields

- Stable change ID, title, author role, date and change class.
- Files, model/data/configuration versions and hashes affected.
- Intended outcome and rollback reference.
- Risk-register and threat-model impacts.
- Validation plan and results, including failures and skipped checks.
- Required roles, recorded decisions, conditions and expiry where applicable.
- Release effect, production status and unresolved blockers.

An approval must name the role, decision, date and evidence reference. Do not add an approval on
behalf of an absent reviewer.

## Decision rules

- Failed registered safety gates cannot be waived by relabelling them exploratory.
- A champion change requires the locked external evaluation and promotion policy.
- Lowering a coverage, robustness, privacy, security or monitoring gate requires explicit risk
  acceptance with expiry and compensating control.
- Unavailable real-CUDA, container, gateway or external-data evidence must remain `unverified`.
- Release approval does not create production approval; that is a separate scope/high-risk decision.
- Emergency action may reduce service but may not silently expand use or data access.

## Module 17 record

`governance/change-approvals/module17.yaml` truthfully records that Module 17 changes documentation
and governance contracts, not runtime behaviour. It remains `pending_release_approval` with an empty
approval list. This prevents the repository from fabricating independent review and blocks v0.2.0
release until accountable roles record decisions. The clean-clone Module 17 check also found that
the committed Module 11A notebook retains a local Mac path in saved output; publication hygiene is
therefore explicitly failed until Module 18 sanitises and revalidates that notebook.

## Verification and retention

Records are reviewed by automated consistency tests and Git history. Approved records must be
included with release evidence and retained according to an organisation-approved schedule. The
research repository does not yet define a legally reviewed retention period.
