# Data policy

Raw datasets are downloaded into ignored local storage and are never committed. The source is
pinned to a full upstream Git commit and each file is checked against a versioned SHA-256 digest.
The loader fails closed if a local or downloaded file differs.

Run from the repository root:

```bash
python scripts/prepare_banking77.py
```

After the first successful download, reproduce the manifest without network access:

```bash
python scripts/prepare_banking77.py --offline
```

The committed manifest records file hashes, exact source-row indices, class counts, data-policy
settings and split-integrity checks. It intentionally contains no banking messages. The official
test file remains unchanged; normalized train/test duplicates are removed only from the training
side and recorded as hashed quarantine events.

Small synthetic fixtures may be committed when they contain no customer or confidential data.
See [`docs/DATA_CARD.md`](../docs/DATA_CARD.md) for provenance, limitations and the verified audit.

Module 8's `fixtures/synthetic-possible-ood.jsonl` contains 96 authored non-banking messages across
12 domains. It contains no customer data and is not a sample of production traffic. Scenario groups
are kept intact when forming threshold-development and possible-OOD-assessment roles. Results from
this fixture must always be labelled synthetic.
