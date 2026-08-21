# AI incident-response process

## Scope

This process covers model, data, privacy, security, availability, governance and supply-chain
events involving the Governed Banking Intent Router. It supplements—not replaces—INNETWORK
Technology Limited's legal, cybersecurity, privacy and business-continuity processes.

## Incident triggers

- Customer, credential or redacted text appears in audit, telemetry, logs, reports or Git history.
- A security-sensitive request is routed below the required security path.
- An unapproved model, policy, dataset, dependency or container revision becomes active.
- Authentication, gateway isolation, artifact-hash or audit fail-closed controls are bypassed.
- Unexpected public exposure, real customer data or autonomous use occurs.
- Material drift, repeated confident errors, abnormal routing distribution or review overload occurs.
- A dependency, CI credential or release artifact is compromised.
- Liveness/readiness, audit persistence or telemetry failures create an operational blind spot.

## Severity

| Severity | Research-preview meaning | Initial action |
|---|---|---|
| SEV-1 | Confirmed exposure of real customer/credential data, public auth bypass, malicious release or customer-impacting autonomous action | Stop service and distribution; activate incident commander immediately |
| SEV-2 | Credible high-impact control failure without confirmed customer impact, or repeated missed security routing | Isolate affected revision and begin containment |
| SEV-3 | Bounded degradation, failed monitoring/audit path, non-exploited vulnerability or significant drift | Restrict use and investigate promptly |
| SEV-4 | Documentation, test or low-impact anomaly with controls intact | Track through normal change process |

These labels do not define statutory notification deadlines. The security/privacy reviewer must
obtain jurisdiction- and contract-specific advice when real organisations or people may be affected.

## Response procedure

1. **Detect and record:** open an incident identifier; record time, reporter, affected versions,
   environment and categorical indicators. Do not copy sensitive message content into tickets.
2. **Triage:** assign severity, incident commander and model/security/privacy participants. Assume
   real data is sensitive until established otherwise.
3. **Contain:** disable the route, block gateway access, revoke/rotate exposed credentials, stop
   artifact distribution and preserve immutable evidence. Use the rollback procedure where safe.
4. **Eradicate:** remove the cause through a reviewed change. Never hot-edit a running model or
   policy because that destroys reproducibility.
5. **Recover:** deploy an approved immutable revision; verify hashes, authentication, privacy,
   policy, audit, health and monitoring checks before restoring research use.
6. **Communicate:** notify accountable roles and any affected external parties through approved
   organisational channels. Public disclosure must be accurate and must not expose sensitive data.
7. **Learn:** complete a blameless post-incident review covering timeline, control performance,
   harms, evidence gaps, risk-register changes and required revalidation.

## Evidence preservation

Preserve commit and image digests, configuration/model hashes, deployment and policy versions,
metadata-only audit records, CI attestations, alerts and access-control events. Restrict access and
record chain of custody. Do not create message hashes as a workaround; short-message hashes can be
re-identifying. Obtain authorised source-system evidence only when required and lawfully approved.

## Closure criteria

An incident closes only when containment is verified, mandatory notifications are resolved,
corrective actions have owners and due dates, risks and threat scenarios are updated, required
validation passes, and the accountable system owner plus security/privacy reviewer approve closure.
SEV-1 and SEV-2 incidents require independent model-risk review before reactivation.

## Exercises

Run a tabletop exercise at least every 180 days while the system is active and before any pilot.
At minimum exercise PII leakage, compromised release, missed security routing and gateway exposure.
