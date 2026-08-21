# Validation and revalidation policy

## Purpose

Validation asks whether the complete system is fit for its stated, bounded use—not merely whether a
classifier has high average accuracy. Verification checks that implementation and evidence match
their specifications. Neither activity creates production approval without a separate accountable
decision.

## Independence and roles

The change author may run tests but cannot be the sole approver for model-risk, control or scope
changes. The model risk reviewer challenges model/data/evaluation evidence; the security/privacy
reviewer challenges threat, privacy and deployment controls; the accountable system owner decides
whether bounded use may proceed. Any unresolved veto blocks release.

## Validation layers

| Layer | Minimum evidence |
|---|---|
| Software verification | Lint, prospective formatting, scoped static typing, unit/integration/policy/privacy tests and safety-module coverage floors |
| Data verification | Pinned provenance/licence, hashes, split integrity, duplicate/leakage checks and no unapproved real data |
| Model evaluation | Multi-seed per-intent metrics, confusion analysis, calibration, selective risk, possible-OOD and security-intent results |
| Robustness | Versioned typo, speech, paraphrase, ambiguity, multi-intent, code-switching, PII, manipulation, non-banking and security cases |
| System validation | Authentication, size/schema controls, redaction-before-inference, deterministic routing, audit fail-closed behaviour and privacy-safe telemetry |
| Operational validation | Real target-device/container, gateway, durable audit/telemetry, capacity, failure recovery, rollback and human-review exercises |
| Independent confirmation | New locked representative external data assessed after model, thresholds and policy are frozen |

## Release gates

Every research-preview release requires clean-clone tests, publication hygiene, package/container
checks, dependency/secret/security scans, complete cards/registers and an approved change record.
Skipped checks and unavailable environments must be visible.

Production or pilot approval additionally requires:

- lawfully sourced representative data with domain, privacy and security approval;
- an untouched locked external comparison across seeds 17, 42 and 73;
- champion-aligned service loading from an immutable registry reference;
- passed classification, security-intent, calibration, selective-risk, privacy and routing gates;
- real target infrastructure, identity gateway, container and rollback evidence;
- approved monitoring baselines, reviewer capacity and incident exercises; and
- independent model-risk, security/privacy and accountable-owner decisions.

The existing BANKING77 test set, validation pools and synthetic fixtures cannot satisfy independent
confirmation.

## Revalidation triggers

Full or targeted revalidation is mandatory after:

- model weights, architecture, tokenizer, feature extraction or champion decision changes;
- training/evaluation data, labels, split logic, normalisation or fixture changes;
- calibration, uncertainty, threshold, risk tier, queue or policy changes;
- privacy detectors, audit schema, telemetry allowlist or retention changes;
- API, authentication, gateway, runtime, dependency, container or infrastructure changes;
- new language, jurisdiction, user group, data source, public exposure, tenancy or intended use;
- drift/monitoring threshold breach, material reviewer overrides, incident or near miss;
- a relevant vulnerability, upstream model/data revision or licence change; or
- 90 days without governance review while the system remains active.

## Data and test discipline

Register hypotheses, seeds, metrics, gates and dataset roles before evaluation. Do not tune on an
assessment or official test set after observing its results. Preserve negative results. Report
confidence intervals where claims depend on model comparison. Synthetic evidence must remain
labelled and cannot support prevalence, fairness or production claims.

For real data, approve minimisation, purpose, lawful basis, consent/notice where applicable, access,
retention, deletion, security, representativeness and prohibited attributes before access. This
repository must not contain customer records.

## Validation report and decision

Each validation report must identify commits, data/model/configuration hashes, environment,
hardware, dependency versions, seed, exclusions, missing artifacts, results, confidence intervals,
failed gates, limitations and a reproducibility procedure. The approval record must state
`approved`, `approved_with_conditions`, `rejected` or `pending`; silence is not approval.

## Current decision

The project is validated only for bounded local research and open-source inspection. Failed
uncertainty and robustness gates, absent representative data, service/champion mismatch and
unverified deployment infrastructure prohibit a production-stable label.
