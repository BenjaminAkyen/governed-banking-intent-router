"""Hash-bound LoRA inference on a real Module 11 accelerator profile."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from governed_banking.accelerator import (
    AcceleratorMetadata,
    empty_accelerator_cache,
    seed_accelerator,
    select_accelerator,
    synchronize_accelerator,
)
from governed_banking.calibration import probabilities_from_logits
from governed_banking.frozen_baseline import resolve_model_snapshot
from governed_banking.inference import LoRAPredictorConfig
from governed_banking.lora_baseline import build_base_sequence_classifier
from governed_banking.runtime_evidence import RuntimeProfile


@dataclass(frozen=True)
class PortablePrediction:
    """A calibrated prediction with the full vector required for parity evidence."""

    predicted_intent: str
    predicted_index: int
    probabilities: tuple[float, ...]
    model_seed: int
    uncertainty_signal: str
    uncertainty_score: float
    model_artifact_sha256: str


class PortableLoRAPredictor:
    """Run the immutable Module 10 model source on an explicit Module 11 device."""

    def __init__(
        self,
        source_config: LoRAPredictorConfig,
        runtime_profile: RuntimeProfile,
    ) -> None:
        if (
            runtime_profile.device_preference == "mps"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        ):
            raise RuntimeError("Module 11 MPS inference prohibits CPU operator fallback")

        device, runtime = select_accelerator(
            runtime_profile.device_preference,
            cuda_device_index=runtime_profile.cuda_device_index,
        )
        if runtime_profile.require_accelerator and device.type == "cpu":
            raise RuntimeError("runtime profile requires a real accelerator")
        if runtime_profile.device_preference != "auto" and device.type != (
            runtime_profile.device_preference
        ):
            raise RuntimeError("explicit runtime profile changed backend")

        seed_accelerator(source_config.seed, device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # This validates the historical MPS registration and every bound artifact hash.
        # Device selection above is intentionally separate and does not alter that evidence.
        experiment, label_names = source_config.validate_sources()
        snapshot = resolve_model_snapshot(
            experiment.encoder,
            cache_directory=source_config.model_cache_directory,
            offline=source_config.offline_only,
        )
        base_model = build_base_sequence_classifier(
            snapshot,
            label_names=label_names,
            attention_implementation=experiment.attention_implementation,
        )
        model = PeftModel.from_pretrained(
            base_model,
            source_config.checkpoint_directory,
            is_trainable=False,
        )
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot.path,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
        )

        self._source_config = source_config
        self._runtime_profile = runtime_profile
        self._experiment = experiment
        self._label_names = label_names
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._runtime = runtime
        self._lock = threading.Lock()

    @property
    def runtime(self) -> AcceleratorMetadata:
        return self._runtime

    @property
    def runtime_profile(self) -> RuntimeProfile:
        return self._runtime_profile

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._label_names

    def predict(self, redacted_text: str) -> PortablePrediction:
        if not isinstance(redacted_text, str) or not redacted_text.strip():
            raise ValueError("redacted_text must be a non-blank string")
        with self._lock, torch.inference_mode():
            encoded = self._tokenizer(
                redacted_text,
                truncation=True,
                max_length=self._experiment.encoder.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            logits = self._model(**inputs).logits
            synchronize_accelerator(self._device)
            host_logits = logits.detach().to(device="cpu", dtype=torch.float64).numpy()

        values = probabilities_from_logits(host_logits, self._source_config.temperature)[0]
        probabilities = tuple(float(value) for value in values)
        if len(probabilities) != len(self._label_names):
            raise RuntimeError("model probability vector differs from the registered taxonomy")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise RuntimeError("model produced an invalid probability vector")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError("model probability vector does not sum to one")

        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        uncertainty_score = probabilities[predicted_index]
        return PortablePrediction(
            predicted_intent=self._label_names[predicted_index],
            predicted_index=predicted_index,
            probabilities=probabilities,
            model_seed=self._source_config.seed,
            uncertainty_signal=self._source_config.uncertainty_signal,
            uncertainty_score=uncertainty_score,
            model_artifact_sha256=self._source_config.expected_checkpoint_files_sha256[
                "adapter_model.safetensors"
            ],
        )

    def release_accelerator_cache(self) -> None:
        """Synchronize pending work and release unoccupied backend cache."""

        synchronize_accelerator(self._device)
        empty_accelerator_cache(self._device)
