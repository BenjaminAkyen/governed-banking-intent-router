# Documentation

This directory contains the maintained technical and governance documentation for the Governed
Banking Intent Router. The project is a **v0.2.0 research preview**. It is not approved for
production banking use or real customer data.

## Start here

| Document | Audience | Purpose |
|---|---|---|
| [Architecture](ARCHITECTURE.md) | Engineers, security reviewers | System boundaries, request flow, controls and failure behaviour |
| [Evaluation](EVALUATION.md) | ML engineers, model-risk reviewers | Data separation, model results, calibration, uncertainty, robustness and promotion rules |
| [Operations](OPERATIONS.md) | Platform engineers, service operators | Runtime profiles, API operation, observability, CI and release hygiene |
| [System card](SYSTEM_CARD.md) | Risk owners, technical reviewers | System purpose, components, authority and current limitations |
| [Model card](MODEL_CARD.md) | ML engineers, model-risk reviewers | Model inventory, intended task, results and model-specific limitations |
| [Data card](DATA_CARD.md) | Data owners, reviewers | Provenance, splits, fixtures and data-use restrictions |
| [Claims register](CLAIMS_REGISTER.md) | Reviewers, publishers | Claims that are verified, refuted, unsupported or prohibited |

## Governance and assurance

The [governance index](governance/README.md) links the risk register, threat model, intended-use
boundary, human-oversight procedure, incident response, rollback, monitoring, validation and
change-approval process.

Architecture decisions are preserved as ADRs:

- [ADR 0001 — Apple MPS and LoRA](decisions/0001-mac-mps-and-lora.md)
- [ADR 0002 — Cross-platform accelerator runtime](decisions/0002-cross-platform-accelerators.md)
- [ADR 0003 — Champion–challenger promotion](decisions/0003-champion-challenger-promotion.md)

## Sources of truth

Documentation explains the system; it does not override executable contracts.

| Subject | Canonical source |
|---|---|
| Model registry and promotion gates | `configs/champion_challenger.yaml` |
| Data provenance and hashes | `configs/dataset.yaml`, `data/manifests/` |
| Privacy and routing policy | `configs/privacy.yaml`, `configs/routing_policy.yaml` |
| Runtime and deployment profiles | `configs/runtime/`, `configs/deployment/` |
| Governance boundary | `configs/governance/module17.yaml` |
| Risk records | `governance/risk-register.yaml` |
| Release approval | `governance/change-approvals/` |
| Results | `reports/` and the corresponding executable tests |

If prose conflicts with a versioned configuration, report or test, treat the discrepancy as a
documentation defect and block the affected claim until it is resolved.

## Documentation policy

- Keep one maintained document per concern; do not create a new file for each development stage.
- Preserve experimental detail in reports, notebooks and Git history rather than duplicating it in
  narrative documents.
- Label synthetic, post-test, single-device and unverified evidence at the point of use.
- Link every public metric to a committed report or reproducible command.
- Never convert a failed gate into a positive claim by changing terminology.
- Update the cards, claims register, risk register and approval record when a material boundary
  changes.
- Do not store customer text, credentials, local paths, checkpoints, raw datasets or audit logs in
  documentation or notebook output.
