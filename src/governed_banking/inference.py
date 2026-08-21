"""Offline, hash-bound LoRA-RoBERTa inference for the Module 10 shadow service."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from governed_banking.calibration import probabilities_from_logits
from governed_banking.data import sha256_file, stable_json_sha256, validate_manifest
from governed_banking.device import RuntimeDevice, select_device
from governed_banking.frozen_baseline import resolve_model_snapshot
from governed_banking.lora_baseline import build_base_sequence_classifier, hash_checkpoint_files
from governed_banking.multiseed import MultiSeedExperimentConfig


@dataclass(frozen=True)
class Prediction:
    predicted_intent: str
    model_seed: int
    uncertainty_signal: str
    uncertainty_score: float
    model_artifact_sha256: str


class IntentPredictor(Protocol):
    def predict(self, redacted_text: str) -> Prediction: ...


@dataclass(frozen=True)
class LoRAPredictorConfig:
    seed: int
    device: str
    offline_only: bool
    multiseed_config_path: Path
    expected_multiseed_config_sha256: str
    manifest_path: Path
    expected_manifest_file_sha256: str
    expected_manifest_sha256: str
    checkpoint_directory: Path
    expected_checkpoint_files_sha256: Mapping[str, str]
    calibration_report_path: Path
    expected_calibration_file_sha256: str
    expected_calibration_report_sha256: str
    temperature: float
    uncertainty_signal: str
    uncertainty_status: str
    model_cache_directory: Path

    def validate_sources(self) -> tuple[MultiSeedExperimentConfig, tuple[str, ...]]:
        if self.seed != 42:
            raise ValueError("Module 10 is registered for seed 42")
        if self.device != "mps":
            raise ValueError("Module 10 is registered for MPS inference")
        if self.offline_only is not True:
            raise ValueError("Module 10 model resolution must remain offline-only")
        if self.uncertainty_signal != "max_probability":
            raise ValueError("seed-42 Module 8 signal must remain max_probability")
        if self.uncertainty_status != "experimental_review_only":
            raise ValueError("Module 8 uncertainty status must remain experimental")
        _finite_positive(self.temperature, "temperature")
        _expected_file_hash(
            self.multiseed_config_path,
            self.expected_multiseed_config_sha256,
            "multi-seed configuration",
        )
        experiment = MultiSeedExperimentConfig.from_yaml(self.multiseed_config_path)
        if self.seed not in experiment.seeds or experiment.encoder.device != self.device:
            raise ValueError("service model differs from the multi-seed registration")
        _expected_file_hash(
            self.manifest_path,
            self.expected_manifest_file_sha256,
            "BANKING77 manifest",
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest.get("manifest_sha256") != self.expected_manifest_sha256:
            raise ValueError("BANKING77 manifest content hash differs from service registration")
        labels = tuple(str(label) for label in manifest.get("label_names", []))
        if len(labels) != 77 or len(labels) != len(set(labels)):
            raise ValueError("service requires the registered 77-label taxonomy")
        actual_checkpoint_hashes = hash_checkpoint_files(self.checkpoint_directory)
        if actual_checkpoint_hashes != dict(self.expected_checkpoint_files_sha256):
            raise ValueError("LoRA checkpoint hashes differ from service registration")
        _expected_file_hash(
            self.calibration_report_path,
            self.expected_calibration_file_sha256,
            "calibration report",
        )
        calibration = json.loads(self.calibration_report_path.read_text(encoding="utf-8"))
        calibration_body = dict(calibration)
        report_sha256 = calibration_body.pop("report_sha256", None)
        if report_sha256 != stable_json_sha256(calibration_body):
            raise ValueError("calibration report content hash check failed")
        if report_sha256 != self.expected_calibration_report_sha256:
            raise ValueError("calibration report differs from service registration")
        if calibration.get("seed") != self.seed:
            raise ValueError("calibration report uses a different model seed")
        if not math.isclose(
            float(calibration.get("temperature_fit", {}).get("temperature", math.nan)),
            self.temperature,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("temperature differs from the registered calibration report")
        if (
            calibration.get("extraction", {}).get("checkpoint_files_sha256")
            != actual_checkpoint_hashes
        ):
            raise ValueError("calibration report and service checkpoint differ")
        return experiment, labels


class LoRAPredictor:
    """Single-model offline predictor; a lock serializes MPS access across request threads."""

    def __init__(self, config: LoRAPredictorConfig) -> None:
        experiment, label_names = config.validate_sources()
        snapshot = resolve_model_snapshot(
            experiment.encoder,
            cache_directory=config.model_cache_directory,
            offline=config.offline_only,
        )
        base_model = build_base_sequence_classifier(
            snapshot,
            label_names=label_names,
            attention_implementation=experiment.attention_implementation,
        )
        model = PeftModel.from_pretrained(
            base_model,
            config.checkpoint_directory,
            is_trainable=False,
        )
        device, runtime = select_device(config.device)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot.path,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
        )
        self._config = config
        self._experiment = experiment
        self._label_names = label_names
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._runtime = runtime
        self._lock = threading.Lock()

    @property
    def runtime(self) -> RuntimeDevice:
        return self._runtime

    def predict(self, redacted_text: str) -> Prediction:
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
            logits = self._model(**inputs).logits.detach().cpu().numpy()
        probabilities = probabilities_from_logits(logits, self._config.temperature)[0]
        predicted_index = int(probabilities.argmax())
        uncertainty_score = float(probabilities[predicted_index])
        if not math.isfinite(uncertainty_score) or not 0.0 <= uncertainty_score <= 1.0:
            raise RuntimeError("model produced an invalid uncertainty score")
        return Prediction(
            predicted_intent=self._label_names[predicted_index],
            model_seed=self._config.seed,
            uncertainty_signal=self._config.uncertainty_signal,
            uncertainty_score=uncertainty_score,
            model_artifact_sha256=self._config.expected_checkpoint_files_sha256[
                "adapter_model.safetensors"
            ],
        )


def _expected_file_hash(path: Path, expected: str, name: str) -> None:
    _sha256(expected, f"expected {name} SHA-256")
    if sha256_file(path) != expected:
        raise ValueError(f"{name} file hash differs from service registration")


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_positive(value: float, name: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)
