# Monitoring plan

## Objective

Monitoring must detect safety, privacy, security, performance and governance degradation without
collecting customer text, redacted text, tokens, names, account data, request/correlation IDs or
message hashes.

## Registered telemetry

The service registers request/error counts, latency, model-loading duration, selected device,
model and policy versions, human-review rate, security-escalation rate, redaction-category counts,
uncertainty distribution and routing-distribution change. Attribute values are fixed and
allowlisted in `configs/observability.yaml`.

## Indicators and required response

| Indicator | Research-preview trigger | Response |
|---|---|---|
| Readiness | Any unready active instance | Remove from service; inspect model, audit and configuration load |
| Audit persistence | Any route-time append failure | Reject response and open incident investigation |
| Policy action | Any `suggest_queue` event | SEV-2: stop affected revision; current mode cannot emit it |
| Versions | Unknown or unapproved model/policy/privacy version | Remove revision and verify artifact chain |
| Security routing | Any registered mandatory security case misses security | Stop release/pilot and revalidate |
| Privacy | Prohibited key/value, residual PII canary or customer text in a sink | SEV-1/2 containment depending on exposure |
| Errors | Sustained change from an approved baseline | Investigate by endpoint/status category without text |
| Latency/capacity | Queue rejection or latency outside the approved environment baseline | Reduce load; investigate capacity and dependency health |
| Human review | Workload exceeds documented reviewer capacity | Pause intake; never lower review to recover throughput |
| Routing distribution | Material change from an approved representative baseline | Investigate drift and trigger revalidation |
| Uncertainty | Material shift from an approved representative baseline | Treat as diagnostic; never lower risk automatically |
| Supply chain | Critical/high actionable vulnerability or failed integrity check | Block release and assess active revisions |

Only invariant triggers are fixed above. The project has no representative production baseline, so
numeric traffic, latency, drift and workload thresholds must not be invented from synthetic or
single-Mac evidence. A pilot must preregister them from governed baseline data and capacity tests.

## Dashboards and alerts

Dashboards must separate environment, approved model release and policy version. Alerts must route
to the service operator and the relevant veto role, include only categorical metadata and link to
the runbook. Access to audit and monitoring systems must be role-restricted and itself audited.

## Review cadence

- Per deployment: readiness, version and integrity checks.
- Continuous while running: errors, latency, capacity, audit health and prohibited events.
- Weekly during an approved pilot: model/routing distributions, reviewer workload and overrides.
- Every release and at least every 90 days: monitoring efficacy, alert tests, baselines and risks.
- After incidents or material changes: immediate review and revalidation.

The weekly pilot cadence is dormant until a separately approved pilot exists.

## Privacy validation

Use synthetic canaries to test redaction and sink inspection. Inspect telemetry schemas and sampled
metadata—not customer payloads. The Collector redaction processor is defence in depth; application
allowlisting remains primary. Any request for richer telemetry is a privacy and change-approval
event.

## Current gaps

No real Collector/backend, distributed gateway, durable central audit store, production alerting,
representative drift baseline, reviewer-capacity baseline or operational SLO is verified. These
gaps block production approval.
