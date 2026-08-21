# Configuration policy

Versioned experiment and routing configurations live here. Thresholds and risk mappings must not
be embedded as unexplained constants in notebooks or API code.

`dataset.yaml` pins the BANKING77 source and split policy. `baseline_tfidf.yaml` predeclares the
candidate feature spaces, solver settings, seed and validation-only selection order. Changing an
experiment configuration invalidates its existing evidence artifact.
