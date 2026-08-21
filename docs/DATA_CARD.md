# BANKING77 Data Card

## Intended use in this project

BANKING77 is used as a public benchmark for research into governed intent routing. It supports
controlled comparison of classifiers and routing policies; it is not treated as representative
production traffic or as evidence that a system is safe for a real bank.

## Provenance and immutability

- Upstream repository: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)
- Upstream revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Source directory: `banking_data`
- Expected source rows: 10,003 train and 3,080 test
- Labels: 77
- Local raw files: ignored by Git
- Reproducibility artifact: `data/manifests/banking77-seed-42.json`

The full revision and SHA-256 values in `configs/dataset.yaml` form the acquisition allowlist. A
hash mismatch stops preparation before any rows are parsed.

## Split construction

The loader preserves the official test file exactly. It applies Unicode NFKC normalisation,
case-folding and whitespace collapse solely to identify duplicate groups. Punctuation and lexical
content are not removed.

The policy is:

1. quarantine official-training rows that match a normalised official-test message;
2. quarantine a training group if identical normalised text has conflicting labels;
3. keep remaining duplicate groups together;
4. make a label-stratified validation split over groups, not individual rows;
5. verify zero normalised-text overlap across final train, validation and test splits.

No message text is copied into the committed manifest. Row indices, label counts and hashes are
sufficient to reconstruct the splits from verified local source files.

## Verified seed-42 audit

| Measure | Result |
|---|---:|
| Final training rows | 8,495 |
| Final validation rows | 1,501 |
| Official test rows retained | 3,080 of 3,080 |
| Original train/test overlap groups | 7 |
| Quarantined training rows | 7 |
| Conflicting-label training groups | 0 |
| Final cross-split overlap groups | 0 |

These are data-engineering results, not model-performance results. Their evidence is the committed
manifest and automated tests.

## Known limitations

- Normalised exact matching does not detect semantic paraphrases or near-duplicates.
- Group stratification approximates the requested row fraction because groups have different sizes.
- BANKING77 is English, narrow-domain and historical; demographic and regional representativeness
  cannot be inferred from it.
- Public benchmark labels do not define a bank's operational taxonomy, escalation obligations or
  risk appetite.
- This project does not redistribute the raw data. Users must review upstream terms before use.

## Module 8 synthetic possible-OOD fixture

`data/fixtures/synthetic-possible-ood.jsonl` contains 96 authored messages across 12 non-banking
domains and 48 scenario groups. It contains no customer data. Scenario groups—not individual
messages—are stratified by domain into 48 threshold-development and 48 assessment messages, with
zero group overlap.

This fixture is a controlled challenge set rather than an external dataset. It cannot establish
production OOD detection, reflect unknown-request prevalence or represent real linguistic and
adversarial diversity. Every published result derived from it must be labelled synthetic and use
the term “possible OOD.”

## Module 9 synthetic control fixtures

`data/fixtures/pii-redaction-cases.jsonl` contains 23 authored positive and negative cases covering
the 11 registered structured-PII detectors. Values use documentation ranges and synthetic examples;
they are not customer records. The fixture is deliberately committed because exact expected
redactions must be reproducible, but its text is never copied into audit events or the summary
report.

`data/fixtures/routing-safety-cases.jsonl` contains eight metadata-only policy cases covering high
and low confidence, missing uncertainty, security intent, exposed authentication secret, sensitive
PII, redaction failure and unsupported intent. It contains no message text.

Neither fixture estimates production prevalence, precision, recall, demographic performance or
adversarial robustness. Real evaluation requires lawfully sourced, representative and access-
controlled data with privacy and domain-owner approval.

## Module 10 synthetic API fixture

`data/fixtures/api-shadow-cases.jsonl` contains 12 authored requests for local end-to-end service
checks. It includes synthetic structured identifiers and controlled robustness categories; it
contains no customer data. Only the authentication-secret case registers a mandatory routing
override because the other cases intentionally test the real model without treating its predictions
as ground truth.

The fixture measures integration behaviour and machine-local latency. It does not measure intent
accuracy, production traffic mix, throughput, concurrency capacity or a real attack distribution.
