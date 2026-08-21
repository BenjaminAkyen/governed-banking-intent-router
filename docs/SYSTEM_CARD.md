# System card: Governed Banking Intent Router

## System status

The Governed Banking Intent Router is an open-source, single-organisation research system owned by
INNETWORK Technology Limited. The target release is v0.2.0 research preview. It is not production
approved and must remain in `shadow_review_only` mode.

## Intended outcome

The system studies whether model evaluation, structured-PII minimisation, calibrated confidence,
uncertainty observations, deterministic escalation, human oversight and privacy-safe observability
can make intent-classification research more governable. Its only permitted decision support is an
advisory recommendation to a human-review or security queue.

## Components and responsibilities

| Component | Responsibility | Governing evidence |
|---|---|---|
| Input boundary | Authenticate, validate host/schema/body size and assign request metadata | `src/governed_banking/api.py`, `src/governed_banking/deployment_service.py` |
| Privacy control | Detect and redact registered structured identifiers before inference | `src/governed_banking/privacy.py`, `configs/privacy.yaml` |
| Shadow model | Produce intent probabilities from redacted text | `src/governed_banking/portable_inference.py` |
| Calibration and uncertainty | Scale probabilities and produce diagnostic observations | `src/governed_banking/calibration.py`, `src/governed_banking/uncertainty.py` |
| Deterministic policy | Apply security, privacy, uncertainty and shadow-mode precedence | `src/governed_banking/policy.py`, `configs/routing_policy.yaml` |
| Human oversight | Review every advisory result and retain decision authority | `docs/governance/HUMAN_OVERSIGHT.md` |
| Audit boundary | Persist an allowlisted metadata event or reject the route response | `src/governed_banking/audit.py`, `src/governed_banking/audit_store.py` |
| Observability | Export registered metrics and spans without request identifiers or text | `src/governed_banking/observability.py`, `configs/observability.yaml` |
| Deployment boundary | Enforce readiness, capacity, gateway authentication and immutable rollback | `configs/deployment/`, `deploy/` |
| CI boundary | Test, scan, build and publish time-limited evidence artifacts | `.github/workflows/` |

## Data flow

1. An authenticated caller submits a bounded message.
2. The message exists transiently in process memory while structured identifiers are redacted.
3. Only the redacted representation enters the model.
4. Calibration and uncertainty logic produce diagnostic probability metadata.
5. The deterministic policy routes to human review or security review; it cannot emit a normal
   suggestion in the registered mode.
6. A metadata-only audit event is persisted before a successful response.
7. Privacy-allowlisted metrics and spans may be exported without text, identities, request IDs or
   message-derived hashes.

## Deployment profiles

- Native macOS/MPS: local development, loopback binding and development bearer token.
- Linux CPU container: private origin behind a trusted gateway; runtime execution pending.
- Linux CUDA container: private origin behind a trusted gateway; real-CUDA execution pending.

The gateway and Kubernetes files are templates, not deployed infrastructure. The identity-provider
fields are deliberately unresolved, so they cannot support a production-readiness claim.

## Decision authority and human oversight

The software recommends; people decide. Human reviewers may override any intent or queue. The
router cannot authenticate a customer, change an account, move money, resolve a complaint, decide
fraud, give regulated advice or automatically create training data from reviewer feedback.

Role ownership, reviewer procedure, incident authority and veto rights are defined in
`configs/governance/module17.yaml`.

## Safety properties and failure behaviour

- Redaction failure produces human review.
- Inference failure produces human review.
- Audit persistence failure rejects the response.
- Unsupported or invalid uncertainty metadata cannot lower risk.
- Security intents and exposed authentication secrets override lower-risk signals.
- Artifact and configuration mismatches prevent readiness.
- Capacity exhaustion returns a controlled rejection rather than an unbounded queue.
- Rollback replaces the immutable revision; hot model mutation is prohibited.

These are implemented properties, not proof that all failures or personal data will be detected.

## Evidence status

Real MPS runtime, service and in-memory observability tests have passed on the documented Mac.
Cross-platform CPU CI, CodeQL and container jobs are configured but require GitHub-hosted evidence.
The current system still has eight explicit release blockers in
`configs/governance/module17.yaml`, including representative data, uncertainty, robustness,
champion alignment and deployment-infrastructure gaps.

## External dependencies

The system depends on PyTorch, Transformers, PEFT, scikit-learn, FastAPI, OpenTelemetry, the pinned
`roberta-base` artifact and the BANKING77 source. Their licences and security state require review
at each release. Model and data artefacts are not redistributed by this repository.

## System limitations

See `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`, `docs/governance/RISK_REGISTER.md` and the repository
threat model. No statement in this card supersedes the prohibited uses or creates regulatory,
security or production approval.

## Card maintenance

- Owner: `accountable_system_owner`
- Accountable organisation: INNETWORK Technology Limited
- Version: `module17-research-preview-v1`
- Effective date: 2026-08-21
- Review: every release, material change, relevant incident and at least every 90 days
