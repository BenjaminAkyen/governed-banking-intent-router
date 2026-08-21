# Human-oversight procedure

## Purpose

Human oversight is the decision boundary, not a fallback label. Every v0.2.0 router output is an
advisory signal that must be reviewed by an authorised person before it affects a customer or an
operational queue.

## Roles

| Role | Responsibility | Authority |
|---|---|---|
| Accountable system owner | Own scope, resources and release decision | Stop use and veto release |
| Model risk reviewer | Review model evidence, drift, failures and promotion claims | Reject model or threshold changes |
| Security/privacy reviewer | Review security escalations, PII controls and incidents | Stop service and initiate incident response |
| Service operator | Run approved revision and monitor health | Disable or roll back; cannot alter model/policy silently |
| Human queue reviewer | Inspect the underlying request in the authorised support system | Accept, correct or override the advisory route |

High-risk model, policy, data or use changes require a reviewer independent of the change author.
Role assignments must be made in the operator's access-controlled system before any pilot.

## Review workflow

1. Confirm the caller and source system are authorised; do not copy customer text into this
   repository, telemetry or free-form audit notes.
2. Treat the predicted intent, confidence and uncertainty observation as non-authoritative.
3. For `security_queue`, apply the organisation's independent security triage procedure. Do not
   infer fraud or compromise solely from the model output.
4. For `human_review`, determine the correct queue from the authorised source record and available
   policy—not from confidence alone.
5. Record a categorical outcome: `accepted`, `overridden`, `insufficient_information` or
   `escalated`. Record an allowlisted reason code; do not record message text in router telemetry.
6. Use established customer-authentication and banking procedures outside this router for every
   consequential action.
7. Report recurrent errors, suspected leakage, unsafe confidence or workload pressure through the
   incident and revalidation processes.

## Mandatory escalation

Immediately escalate to the security/privacy reviewer when the request suggests exposed
credentials, a compromised or stolen device/card, unrecognised payment/withdrawal/direct debit, or
when personal data appears in an unauthorised sink. Escalate to the model risk reviewer when a known
security case is not sent to security, the model produces repeated confident errors, or the routing
distribution changes unexpectedly.

## Override and recourse

Reviewers may always override the router. Overrides must not be treated as reviewer failure and may
not enter training automatically. Feedback is quarantined until provenance, access, label quality,
privacy and change approval are complete. Any future affected-user challenge or correction must be
handled through the operator's established complaints and recourse channels; this repository does
not implement those channels.

## Capacity and competence gate

No pilot may begin until the operator documents reviewer training, queue ownership, access control,
coverage hours, response targets, workload limits, absence/continuity arrangements and escalation
contacts. These values are deliberately not invented for the research preview. If reviewers cannot
keep up or independent verification is unavailable, pause the service rather than relax review.

## Oversight monitoring

Monitor review rate, security-escalation rate, override categories, unresolved age, workload and
reviewer agreement using approved systems. Never export customer text, identity, request IDs or
message hashes through router telemetry. Review outcomes are evidence for investigation and
revalidation, not automatic retraining labels.
