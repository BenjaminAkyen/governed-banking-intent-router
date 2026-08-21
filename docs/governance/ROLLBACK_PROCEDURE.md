# Rollback procedure

## Principle

Rollback replaces an immutable process or container revision and its versioned model bundle. Hot
mutation of model, calibration, policy or privacy files is prohibited because it breaks the
evidence chain and can leave workers inconsistent.

## Rollback triggers

- Artifact, configuration or evidence hash mismatch.
- Failed privacy, policy, audit, robustness or validation gate.
- Authentication or private-origin control failure.
- New critical/high vulnerability affecting the active revision.
- Unexpected suggestion action, missed mandatory security escalation or repeated confident error.
- Audit persistence failure or privacy-unsafe telemetry.
- Capacity, readiness or latency degradation outside an approved operating envelope.
- Incident commander or release-veto decision.

## Preconditions

The service operator must have an approved previous revision identified by immutable reference,
its model/configuration checksums, recorded approval, deployment manifest and known limitations.
Container profiles require `GOVERNED_BANKING_RELEASE_ID` and
`GOVERNED_BANKING_ROLLBACK_REFERENCE`; readiness validation must reject an absent or self-referential
rollback reference.

## Procedure

1. Declare the rollback in the incident or change record and stop further rollout.
2. Preserve metadata-only evidence for the failing revision; do not copy customer text.
3. Remove the failing revision from service through the deployment platform.
4. Deploy the previously approved immutable process/container and its bound model bundle.
5. Verify release ID, model hash, runtime profile, policy/privacy/audit/observability configuration
   hashes and approved gateway settings.
6. Confirm `/health/live`, then `/health/ready`; liveness alone is insufficient.
7. Run the registered synthetic privacy, routing and service smoke checks.
8. Confirm audit append and privacy-allowlisted telemetry using synthetic canaries.
9. Restore only the permitted research traffic and monitor for recurrence.
10. Record outcome, timestamps, approvers and any residual risk. Begin root-cause analysis.

## Failure of rollback

If the previous revision is unavailable, fails verification or has the same defect, keep the
service disabled. Do not bypass readiness, hashes, audit or authentication to restore availability.
Use the manual support process outside this router.

## Post-rollback requirements

Rollback is containment, not closure. The failed revision remains prohibited until correction,
revalidation and new approval. Update the risk register, incident review, model/system cards and
monitoring plan when the failure changes assumptions or risk.

## Research-preview limitation

The templates describe the control but do not prove a deployed orchestrator, image registry,
identity gateway or rollback operation. A real rollback exercise is required before any pilot.
