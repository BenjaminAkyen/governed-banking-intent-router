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

The next module may train only on the manifest-selected training indices, tune on validation
indices and evaluate on official test indices after all modelling choices are locked.
