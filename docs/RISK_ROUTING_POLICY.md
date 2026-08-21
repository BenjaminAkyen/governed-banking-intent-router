# Deterministic Risk-Routing Policy

## Current decision

Module 9 operates in `shadow_review_only` mode. It may recommend `security_queue` or
`human_review`; it cannot emit `suggest_queue`.

This restriction is intentional. Module 8's uncertainty experiment failed its registered
operating-point gates. A score above an experimental threshold therefore cannot lower risk or
authorize a normal-queue suggestion.

## Precedence

The policy evaluates controls in a fixed order:

1. an exposed authentication secret routes to `security_queue`;
2. a registered fraud or account-security intent routes to `security_queue`;
3. redaction failure routes to `human_review`;
4. an unsupported intent routes to `human_review`;
5. sensitive structured PII routes to `human_review`;
6. experimental uncertainty is recorded as below, above or invalid;
7. shadow mode routes every remaining case to `human_review`.

Security overrides are not cancelled by missing or malformed uncertainty metadata.

## Module 8 binding

`configs/routing_policy.yaml` pins both the file hash and internal aggregate hash of the failed
Module 8 evidence. It also pins each seed's selected signal and threshold. Loading the policy fails
if those values differ or if the evidence claims that all gates passed.

The values are observational only:

- below threshold: add an experimental-uncertainty review reason;
- at or above threshold: still require review and record that the threshold is not authorized;
- missing, malformed or mismatched signal: fail closed to review.

## Security-intent tier

The initial security tier includes unrecognized card, cash-withdrawal and direct-debit activity;
compromised, lost or stolen cards; lost or stolen phones; forgotten passcodes; and blocked PINs.
This research mapping requires review by a bank's fraud, complaints and operations owners before
deployment.

## Limitations

- BANKING77 labels are not an operational risk taxonomy.
- A predicted security intent can be wrong; `security_queue` is prioritization, not a fraud finding.
- The policy does not authenticate a customer, execute an action or decide liability.
- Review capacity, service-level targets and override procedures remain organizational controls.
- Enabling `suggest_queue` requires a new policy schema and independently approved evidence; it is
  not a runtime flag in the current schema.
