# Configuration policy

Versioned experiment and routing configurations live here. Thresholds and risk mappings must not
be embedded as unexplained constants in notebooks or API code.

`dataset.yaml` pins the BANKING77 source and split policy. `baseline_tfidf.yaml` predeclares the
candidate feature spaces, solver settings, seed and validation-only selection order. Changing an
experiment configuration invalidates its existing evidence artifact.

`frozen_roberta.yaml` pins the encoder revision, extraction policy and classifier search. Its
amendment fields preserve why validation search expanded before test access. The final round is
part of the configuration hash and cannot change without invalidating Module 4 evidence.
