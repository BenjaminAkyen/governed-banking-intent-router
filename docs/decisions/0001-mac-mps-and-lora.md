# ADR 0001: Use Apple MPS and standard LoRA first

- Status: accepted
- Date: 2026-08-20

## Context

The primary development machine is an Apple M4 MacBook Pro with 24 GB unified memory. The project
needs a training path that can be reproduced locally without an NVIDIA GPU.

## Decision

- Use PyTorch MPS when available and CPU for CI or unsupported operations.
- Fail fast when a user explicitly requests unavailable MPS.
- Begin with `roberta-base`, maximum sequence length 128, small batches and gradient accumulation.
- Use standard LoRA before attempting quantised adapter training.
- Do not use CUDA-specific dependencies or disable MPS memory safeguards.
- Record runtime metadata with every experiment.

## Consequences

The workflow remains accessible on the available Mac and CI can exercise non-training controls.
Some operations may differ from CUDA behaviour, and full fine-tuning may be slower or memory-bound.
Those constraints will be measured rather than hidden.
