# NIST AI RMF 1.0 mapping

## Scope and method

This is a project-specific current-state mapping to NIST AI RMF 1.0. It is not a certification,
legal opinion or claim of full conformance. NIST describes the framework and Playbook as voluntary,
and the Playbook is not a checklist. NIST also states that AI RMF 1.0 is being revised; this mapping
therefore records the referenced version and must be revisited after a revision.

Primary references:

- [NIST AI RMF 1.0](https://doi.org/10.6028/NIST.AI.100-1)
- [NIST AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/)
- [NIST AI Resource Center](https://airc.nist.gov/)

Status means: **implemented** (repository evidence exists), **partial** (some evidence exists but an
operational or independent element is missing), or **gap** (required evidence does not exist).

## GOVERN

| Outcome | Status | Project evidence and remaining work |
|---|---|---|
| GOVERN 1.1 legal/regulatory requirements | Gap | Prohibited-use and privacy boundaries exist, but no jurisdiction-specific banking/privacy/legal assessment is approved. |
| GOVERN 1.2 trustworthy-AI policies | Partial | System cards plus risk, validation, monitoring, incident and change procedures exist; operational adoption is untested. |
| GOVERN 1.3 risk tolerance | Implemented for research | `governance/risk-register.yaml` defines scoring and prohibits accepting high/critical residual risk for production. |
| GOVERN 1.4 transparent risk documentation | Implemented | Claims register, evidence cards, limitations and machine-readable governance contract. |
| GOVERN 1.5 monitoring and periodic review | Partial | Cadence, indicators, incident and override procedures exist; live backend, alerts and pilot baselines are absent. |
| GOVERN 1.6 AI inventory | Partial | System/model/data cards and champion registry inventory project artifacts; no organisation-wide inventory exists. |
| GOVERN 2.1 roles and responsibility | Partial | Role authority and vetoes are defined; people and independent reviewers are not yet assigned. |
| GOVERN 3.2 workforce diversity/expertise | Gap | No multidisciplinary banking, legal, privacy, customer-support or affected-community review is evidenced. |
| GOVERN 5.1 organisational policies | Partial | Repository procedures exist but are not yet integrated into wider organisational policy. |
| GOVERN 6.1 third-party risk | Partial | Notices, SBOM, dependency review and pinned sources exist; supplier monitoring and signed artifacts remain open. |

## MAP

| Outcome | Status | Project evidence and remaining work |
|---|---|---|
| MAP 1.1 intended purpose | Implemented | System boundary and intended/prohibited-use statements restrict the system to advisory research. |
| MAP 1.5 organisational risk tolerance | Implemented for research | Production blockers and release-veto rules are explicit. |
| MAP 2.1 context and affected actors | Partial | Operator and reviewer context is documented; real bank processes and affected customers were not studied. |
| MAP 2.2 system categorisation | Implemented | Research preview, single-organisation, shadow-review classification is machine-readable. |
| MAP 3.1 benefits and costs | Partial | Research benefits and failure consequences are described; no formal impact assessment or business case exists. |
| MAP 4.1 impacts and likelihoods | Implemented for current scope | Risk register and threat model score concrete model, privacy, security and human-factor harms. |
| MAP 5.1 stakeholder engagement | Gap | No affected-customer, banking-operations, complaints, accessibility or regulatory engagement is evidenced. |

## MEASURE

| Outcome | Status | Project evidence and remaining work |
|---|---|---|
| MEASURE 1.1 appropriate methods | Partial | Registered multi-seed, calibration, uncertainty, robustness and system tests exist; representative external methods remain missing. |
| MEASURE 1.3 independent assessment | Gap | Development and assessment are not institutionally independent; external evaluation is required. |
| MEASURE 2.1 evaluation metrics | Implemented for research | Macro-F1, per-intent errors, calibration, risk/coverage, possible-OOD, privacy and routing metrics are recorded. |
| MEASURE 2.3 performance criteria | Implemented | Preregistered gates preserve failures instead of retrospectively changing thresholds. |
| MEASURE 2.5 validity and reliability | Partial | Seeds and intervals improve reliability, but benchmark/test reuse and lack of representative data limit validity. |
| MEASURE 2.7 security and resilience | Partial | Threat, CI, privacy, capacity and rollback controls exist; penetration, gateway and recovery evidence is missing. |
| MEASURE 2.10 privacy risk | Partial | Synthetic redaction/audit tests and minimisation exist; real-data DLP and privacy assessment are absent. |
| MEASURE 3.1 test results documented | Implemented | Hash-bound reports, manifests, cards and claims register include negative results and limitations. |
| MEASURE 4.1 measurement feedback | Partial | Revalidation triggers connect measurements to governance; no live representative feedback loop exists. |

## MANAGE

| Outcome | Status | Project evidence and remaining work |
|---|---|---|
| MANAGE 1.1 proceed/no-proceed decision | Implemented | Local research may continue; production approval is explicitly denied. |
| MANAGE 1.2 treatment prioritisation | Implemented | Risk register separates release blockers, treatments, owners and residual scores. |
| MANAGE 1.3 risk response | Partial | Avoid, mitigate, monitor and rollback responses are defined; many planned treatments are not implemented. |
| MANAGE 2.1 treatment resources | Gap | Reviewer capacity, operations staffing, budgets and live infrastructure are not assigned. |
| MANAGE 2.2 treatment mechanisms | Partial | Human review, fail-closed audit, immutable rollback and CI controls exist; production mechanisms are unverified. |
| MANAGE 2.3 recovery | Partial | Incident and rollback procedures exist; no real container/gateway recovery exercise has passed. |
| MANAGE 3.1 third-party risks | Partial | Dependency scanning and notices exist; supplier response and signed provenance are outstanding. |
| MANAGE 4.1 post-deployment monitoring | Gap for production | Telemetry schema exists, but no production deployment, representative baseline or reviewed SLO exists. |
| MANAGE 4.2 incident response | Partial | Procedure and exercise cadence exist; roles/contacts and completed exercises are absent. |
| MANAGE 4.3 decommissioning | Partial | Stop/rollback is documented; retention and final disposal require organisational/legal approval. |

## Priority actions

1. Assign independent accountable roles and approve or reject the pending change record.
2. Complete jurisdiction-specific legal, privacy and banking-domain assessments before real data.
3. Lock and independently evaluate a representative external dataset.
4. Resolve uncertainty, robustness, security-routing and champion-alignment blockers.
5. Verify the real gateway, CPU/CUDA containers, durable audit/telemetry and rollback exercise.
6. Establish reviewer capacity, affected-stakeholder engagement and operational monitoring baselines.
7. Revisit this crosswalk when NIST publishes the revised AI RMF.
