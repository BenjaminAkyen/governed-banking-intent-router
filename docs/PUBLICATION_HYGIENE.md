# Publication hygiene

## Purpose

This gate prevents local machine details and runtime artifacts from becoming part of a public
release. It does not certify that a repository is free of every possible secret or licensing risk.

## Registered checks

Run from any directory inside the checkout:

```bash
python scripts/check_publication_hygiene.py --execute-notebook-setup
```

The checker:

1. parses every committed or unignored Jupyter notebook;
2. rejects saved macOS, Linux-user, Windows-user and VS Code resource paths;
3. checks common high-confidence credential signatures in the public tree and Git patch history;
4. rejects tracked virtual environments, raw data, checkpoints, model weights and logs;
5. verifies the required ignore rules;
6. executes only the setup cell of each notebook from the nested notebook directory; and
7. confirms each setup cell discovers the current repository root rather than a named parent
   directory.

The setup-cell check does not rerun training, test evaluation or service benchmarks. Published
experiment outputs remain historical evidence and must not be silently regenerated during a
hygiene pass.

## Storage boundary

| Material | Public repository policy |
|---|---|
| Source, tests, configurations and small evidence reports | Commit |
| Notebooks | Commit after path and output inspection |
| Raw BANKING77 files | Do not commit |
| Pretrained snapshots and trained checkpoints | Do not commit |
| Local audit events | Do not commit |
| Virtual environments and caches | Do not commit |
| Credentials and environment files | Do not commit |

See [Third-party notices](../THIRD_PARTY_NOTICES.md) for the pinned dataset and model terms.
