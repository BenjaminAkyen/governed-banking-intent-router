# Privacy and Audit Control Card

## Purpose and current decision

Module 9 adds pre-inference structured-PII redaction and a metadata-only audit contract for the
research router. The controls passed their registered synthetic implementation checks. They are
approved only for continued local shadow-mode development—not production processing, compliance
claims or automated customer routing.

## Privacy boundary

The original message exists only in transient application memory. Detection returns offsets and
PII type names; matched values are not retained in finding objects. The classifier-facing copy uses
fixed tokens such as `[EMAIL]` and `[PAYMENT_CARD]`. A mandatory second scan fails closed if a
registered detector still matches the redacted copy.

The registered recognizers cover authentication-secret assignments, email addresses, GB-format
IBANs with checksum validation, National Insurance number formats, Luhn-valid payment-card numbers,
context-labelled UK sort codes and bank-account numbers, UK phone forms, IPv4 addresses,
context-labelled dates of birth and UK postcodes. Input is limited to 4,096 characters and null
bytes are rejected.

This is deliberately bounded pattern matching. It does not reliably detect names, addresses beyond
the registered UK postcode form, customer IDs, semantic references, secrets without a registered
label, obfuscation or every international identifier. False positives and false negatives remain
possible.

## Audit event contract

Only exact allowlisted fields are accepted:

- UUID4 event identifier and UTC timestamp;
- model artifact hash, seed, predicted intent and experimental uncertainty observation;
- privacy-policy version and hash, redaction outcome, PII type counts and coarse input-size bucket;
- routing-policy version and hash, shadow mode, action, queue and ordered reason codes.

Original text, redacted text, prompts, responses, payloads, message hashes, exact input lengths and
free-form extensions are not fields in the schema. Message hashes are prohibited because short,
low-entropy support requests can be guessed and reidentified.

The demonstration JSONL sink uses a fixed configuration path, refuses symbolic-link targets, caps
event and read sizes, checks file ownership and type, appends one canonical JSON event per line,
and enforces `0700` directory and `0600` file modes.

## Verified synthetic evidence

| Check | Result |
|---|---:|
| Exact PII expectations | 23/23 matched |
| Registered detector classes exercised | 11/11 |
| Exact routing expectations | 8/8 matched |
| Audit events validated after persistence | 24/24 |
| Original/redacted fixture values found in serialized events | 0 |
| Suggestion actions | 0 |
| Sink permissions | `0700` / `0600` |

Evidence is stored in `reports/governance/module9-controls.json` and reproduced by
`scripts/run_governance_controls.py`. The run performs no model inference and accesses neither
BANKING77 nor its official test split.

## Relationship to uncertainty

The routing policy pins Module 8's failed aggregate and its per-seed signals and thresholds. Scores
may add an experimental review reason, but cannot lower risk or authorize `suggest_queue`. Security
intent and exposed authentication-secret overrides take precedence; malformed or missing metadata
fails closed to human review.

## Production gaps

- Evaluate detection on lawfully sourced, representative multilingual and adversarial traffic.
- Replace or layer pattern matching with approved DLP and secrets controls.
- Define lawful basis, purpose limitation, retention, deletion and data-subject procedures.
- Use a centralized append-only service with authentication, authorization, encryption, integrity
  protection, monitoring and tested incident response.
- Review the risk taxonomy and queues with fraud, operations, complaints, privacy and security
  owners.
- Revalidate end to end after API, model, policy or configuration changes.

Until those gaps are closed, these controls are engineering evidence—not a certification or claim
of regulatory compliance.
