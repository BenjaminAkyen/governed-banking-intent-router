# Third-party notices

The repository's [MIT License](LICENSE) applies to this project's code and original materials. It
does not replace the terms attached to third-party datasets, pretrained models or software
dependencies.

## BANKING77

- Source: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)
- Pinned revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Upstream licence at that revision: [Creative Commons Attribution 4.0 International](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE)
- Project handling: raw dataset files are acquired by the user, hash-verified locally and excluded
  from Git.

The upstream repository asks users of BANKING77 to cite:

> Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić. “Efficient
> Intent Detection with Dual Sentence Encoders.” Proceedings of the 2nd Workshop on NLP for
> Conversational AI, ACL 2020.

Any publication or redistribution must preserve the required attribution and link to the CC BY 4.0
licence. The repository does not redistribute the raw BANKING77 train or test files.

## RoBERTa base

- Model: [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)
- Pinned revision: `e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- Licence recorded by the model repository: MIT
- Project handling: the pretrained snapshot and locally trained adapters are excluded from Git;
  committed evidence records their identifiers and hashes.

Before publishing adapter weights, a release owner must verify the base-model notice, BANKING77
attribution, generated model card and the exact files included in that release. Dependency licences
remain governed by their respective packages.
