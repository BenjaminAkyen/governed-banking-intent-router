"""Immutable BANKING77 acquisition and leakage-resistant split construction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml
from sklearn.model_selection import train_test_split

NORMALIZATION_VERSION = "unicode_nfkc_casefold_whitespace_v1"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    """Validated source and split policy loaded from versioned YAML."""

    name: str
    repository_slug: str
    commit: str
    subdirectory: str
    files: Mapping[str, str]
    expected_sha256: Mapping[str, str]
    expected_train_rows: int
    expected_test_rows: int
    expected_label_count: int
    validation_fraction: float
    seed: int
    normalization: str

    @classmethod
    def from_yaml(cls, path: Path) -> DatasetConfig:
        """Load a dataset configuration and reject ambiguous values."""

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("dataset configuration must be a mapping")

        source = _mapping(raw, "source")
        expected = _mapping(raw, "expected")
        split = _mapping(raw, "split")
        files = _mapping(source, "files")
        expected_sha256 = _mapping(source, "sha256")
        required_files = {"categories", "official_train", "official_test"}
        if set(files) != required_files:
            raise ValueError(f"source.files must contain exactly {sorted(required_files)}")
        if set(expected_sha256) != required_files:
            raise ValueError(f"source.sha256 must contain exactly {sorted(required_files)}")

        commit = str(source["commit"])
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("source.commit must be a full 40-character Git commit")

        for filename in files.values():
            if Path(str(filename)).name != str(filename):
                raise ValueError("source filenames cannot contain directory traversal")
        for digest in expected_sha256.values():
            if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
                raise ValueError(
                    "source SHA-256 values must be 64 lowercase hexadecimal characters"
                )

        validation_fraction = float(split["validation_fraction"])
        if not 0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if split["normalization"] != NORMALIZATION_VERSION:
            raise ValueError(f"normalization must be {NORMALIZATION_VERSION}")

        return cls(
            name=str(raw["name"]),
            repository_slug=str(source["repository_slug"]),
            commit=commit,
            subdirectory=str(source["subdirectory"]).strip("/"),
            files={str(key): str(value) for key, value in files.items()},
            expected_sha256={str(key): str(value) for key, value in expected_sha256.items()},
            expected_train_rows=int(expected["official_train_rows"]),
            expected_test_rows=int(expected["official_test_rows"]),
            expected_label_count=int(expected["label_count"]),
            validation_fraction=validation_fraction,
            seed=int(split["seed"]),
            normalization=str(split["normalization"]),
        )

    def source_url(self, logical_name: str) -> str:
        """Return the raw URL pinned to the configured Git commit."""

        filename = self.files[logical_name]
        return (
            "https://raw.githubusercontent.com/"
            f"{self.repository_slug}/{self.commit}/{self.subdirectory}/{filename}"
        )


@dataclass(frozen=True)
class BankingRecord:
    """A source row whose index remains stable within its official CSV."""

    source_split: str
    source_index: int
    text: str
    category: str

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)


@dataclass(frozen=True)
class QuarantinedRow:
    """A training row excluded before the development split."""

    source_index: int
    reason: str
    normalized_text_sha256: str


@dataclass(frozen=True)
class PreparedSplits:
    """Leakage-resistant row selections derived from official source files."""

    train: tuple[BankingRecord, ...]
    validation: tuple[BankingRecord, ...]
    test: tuple[BankingRecord, ...]
    quarantined_train: tuple[QuarantinedRow, ...]
    label_names: tuple[str, ...]
    original_cross_split_overlap_groups: int
    original_conflicting_train_groups: int


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def normalize_text(text: str) -> str:
    """Normalise only case, Unicode representation and whitespace for leakage checks."""

    normalised = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalised.split())


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def download_source(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    timeout: int = 60,
    force: bool = False,
) -> Path:
    """Download a commit-pinned source file using an atomic replacement."""

    if destination.exists() and not force:
        _verify_source_hash(destination, expected_sha256)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "governed-banking-intent-router/0.1"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - pinned HTTPS URL
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        _verify_source_hash(temporary_path, expected_sha256)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _verify_source_hash(path: Path, expected_sha256: str) -> None:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"source hash mismatch for {path.name}: expected {expected_sha256}, got {observed}"
        )


def read_categories(path: Path, *, expected_count: int) -> tuple[str, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(value, str) and value for value in raw):
        raise ValueError("categories.json must contain a non-empty list of strings")
    categories = tuple(raw)
    if len(categories) != expected_count or len(set(categories)) != expected_count:
        raise ValueError("category count or uniqueness check failed")
    return categories


def read_banking_csv(
    path: Path, *, source_split: str, expected_rows: int | None = None
) -> tuple[BankingRecord, ...]:
    records: list[BankingRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"unexpected CSV columns in {path}: {reader.fieldnames}")
        for source_index, row in enumerate(reader):
            text = (row.get("text") or "").strip()
            category = (row.get("category") or "").strip()
            if not text or not category:
                raise ValueError(f"blank text or category at {source_split} row {source_index}")
            records.append(BankingRecord(source_split, source_index, text, category))

    if expected_rows is not None and len(records) != expected_rows:
        raise ValueError(
            f"{source_split} row-count check failed: expected {expected_rows}, got {len(records)}"
        )
    return tuple(records)


def _indices_by_normalized_text(records: Iterable[BankingRecord]) -> dict[str, list[BankingRecord]]:
    groups: dict[str, list[BankingRecord]] = defaultdict(list)
    for record in records:
        groups[record.normalized_text].append(record)
    return dict(groups)


def prepare_splits(
    official_train: Iterable[BankingRecord],
    official_test: Iterable[BankingRecord],
    label_names: Iterable[str],
    *,
    seed: int,
    validation_fraction: float,
) -> PreparedSplits:
    """Build group-stratified splits while leaving official test rows untouched."""

    train_records = tuple(official_train)
    test_records = tuple(official_test)
    labels = tuple(label_names)
    expected_labels = set(labels)
    if not train_records or not test_records:
        raise ValueError("official train and test files must both be non-empty")
    if set(record.category for record in train_records) != expected_labels:
        raise ValueError("official training labels do not match categories.json")
    if set(record.category for record in test_records) != expected_labels:
        raise ValueError("official test labels do not match categories.json")

    train_groups = _indices_by_normalized_text(train_records)
    test_groups = _indices_by_normalized_text(test_records)
    cross_split_keys = set(train_groups) & set(test_groups)
    conflicting_keys = {
        key for key, records in train_groups.items() if len({row.category for row in records}) > 1
    }

    quarantined: list[QuarantinedRow] = []
    excluded_indices: set[int] = set()
    for key in sorted(cross_split_keys):
        for row in train_groups[key]:
            excluded_indices.add(row.source_index)
            quarantined.append(
                QuarantinedRow(
                    row.source_index,
                    "duplicates_official_test",
                    sha256_bytes(key.encode()),
                )
            )
    for key in sorted(conflicting_keys - cross_split_keys):
        for row in train_groups[key]:
            excluded_indices.add(row.source_index)
            quarantined.append(
                QuarantinedRow(
                    row.source_index,
                    "conflicting_training_labels",
                    sha256_bytes(key.encode()),
                )
            )

    clean_records = tuple(
        record for record in train_records if record.source_index not in excluded_indices
    )
    clean_groups = _indices_by_normalized_text(clean_records)
    group_keys = sorted(
        clean_groups,
        key=lambda key: min(record.source_index for record in clean_groups[key]),
    )
    group_labels = [clean_groups[key][0].category for key in group_keys]
    development_keys, validation_keys = train_test_split(
        group_keys,
        test_size=validation_fraction,
        random_state=seed,
        stratify=group_labels,
    )

    development_key_set = set(development_keys)
    validation_key_set = set(validation_keys)
    development = tuple(
        row for row in clean_records if row.normalized_text in development_key_set
    )
    validation = tuple(row for row in clean_records if row.normalized_text in validation_key_set)

    split_label_sets = {
        "train": {row.category for row in development},
        "validation": {row.category for row in validation},
        "test": {row.category for row in test_records},
    }
    for split_name, observed_labels in split_label_sets.items():
        if observed_labels != expected_labels:
            missing = sorted(expected_labels - observed_labels)
            raise ValueError(f"{split_name} is missing labels: {missing}")

    train_keys = {row.normalized_text for row in development}
    validation_keys_final = {row.normalized_text for row in validation}
    test_keys = set(test_groups)
    has_final_overlap = (
        bool(train_keys & validation_keys_final)
        or bool(train_keys & test_keys)
        or bool(validation_keys_final & test_keys)
    )
    if has_final_overlap:
        raise AssertionError("final split construction contains normalized-text leakage")

    return PreparedSplits(
        train=development,
        validation=validation,
        test=test_records,
        quarantined_train=tuple(sorted(quarantined, key=lambda item: item.source_index)),
        label_names=labels,
        original_cross_split_overlap_groups=len(cross_split_keys),
        original_conflicting_train_groups=len(conflicting_keys),
    )


def _duplicate_group_count(records: Iterable[BankingRecord]) -> int:
    return sum(1 for rows in _indices_by_normalized_text(records).values() if len(rows) > 1)


def _label_distribution(
    records: Iterable[BankingRecord], label_names: Iterable[str]
) -> dict[str, int]:
    counts = Counter(record.category for record in records)
    return {label: counts[label] for label in label_names}


def build_manifest(
    config: DatasetConfig,
    source_paths: Mapping[str, Path],
    splits: PreparedSplits,
) -> dict[str, Any]:
    """Create a text-free manifest that fixes the exact rows used by each split."""

    split_indices = {
        "train": [row.source_index for row in splits.train],
        "validation": [row.source_index for row in splits.validation],
        "test": [row.source_index for row in splits.test],
    }
    quarantined = [
        {
            "source_index": row.source_index,
            "reason": row.reason,
            "normalized_text_sha256": row.normalized_text_sha256,
        }
        for row in splits.quarantined_train
    ]
    sources: dict[str, Any] = {}
    for logical_name, path in source_paths.items():
        sources[logical_name] = {
            "filename": path.name,
            "url": config.source_url(logical_name),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": config.name,
        "source_repository": f"https://github.com/{config.repository_slug}",
        "source_commit": config.commit,
        "sources": sources,
        "policy": {
            "seed": config.seed,
            "validation_fraction": config.validation_fraction,
            "normalization": config.normalization,
            "group_stratification": True,
            "official_test_untouched": True,
        },
        "label_names": list(splits.label_names),
        "splits": {},
        "quarantined_train": quarantined,
        "integrity": {
            "original_train_test_overlap_groups": splits.original_cross_split_overlap_groups,
            "original_conflicting_training_groups": splits.original_conflicting_train_groups,
            "final_train_validation_overlap_groups": 0,
            "final_train_test_overlap_groups": 0,
            "final_validation_test_overlap_groups": 0,
            "train_duplicate_groups": _duplicate_group_count(splits.train),
            "validation_duplicate_groups": _duplicate_group_count(splits.validation),
            "test_duplicate_groups": _duplicate_group_count(splits.test),
        },
        "contains_message_text": False,
    }
    records_by_split = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for split_name, records in records_by_split.items():
        indices = split_indices[split_name]
        manifest["splits"][split_name] = {
            "source": "official_test" if split_name == "test" else "official_train",
            "count": len(records),
            "source_indices": indices,
            "source_indices_sha256": stable_json_sha256(indices),
            "label_distribution": _label_distribution(records, splits.label_names),
        }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    return manifest


def write_manifest(manifest: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def acquire_and_prepare(
    config_path: Path,
    raw_directory: Path,
    manifest_path: Path,
    *,
    offline: bool = False,
    force_download: bool = False,
) -> dict[str, Any]:
    """Acquire pinned sources, create clean splits and write the manifest."""

    config = DatasetConfig.from_yaml(config_path)
    source_paths = {
        logical_name: raw_directory / filename for logical_name, filename in config.files.items()
    }
    for logical_name, destination in source_paths.items():
        if offline:
            if not destination.exists():
                raise FileNotFoundError(f"offline source is missing: {destination}")
            _verify_source_hash(destination, config.expected_sha256[logical_name])
        else:
            download_source(
                config.source_url(logical_name),
                destination,
                expected_sha256=config.expected_sha256[logical_name],
                force=force_download,
            )

    label_names = read_categories(
        source_paths["categories"], expected_count=config.expected_label_count
    )
    official_train = read_banking_csv(
        source_paths["official_train"],
        source_split="official_train",
        expected_rows=config.expected_train_rows,
    )
    official_test = read_banking_csv(
        source_paths["official_test"],
        source_split="official_test",
        expected_rows=config.expected_test_rows,
    )
    splits = prepare_splits(
        official_train,
        official_test,
        label_names,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
    )
    manifest = build_manifest(config, source_paths, splits)
    write_manifest(manifest, manifest_path)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail if a generated manifest violates the project's data contract."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    if manifest.get("contains_message_text") is not False:
        raise ValueError("manifest must not contain message text")
    policy = manifest.get("policy", {})
    if policy.get("official_test_untouched") is not True:
        raise ValueError("official test must remain untouched")
    integrity = manifest.get("integrity", {})
    final_overlap_keys = (
        "final_train_validation_overlap_groups",
        "final_train_test_overlap_groups",
        "final_validation_test_overlap_groups",
    )
    if any(integrity.get(key) != 0 for key in final_overlap_keys):
        raise ValueError("manifest reports final split leakage")
    expected_hash = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("manifest content hash check failed")


def load_manifest_split(
    config_path: Path,
    raw_directory: Path,
    manifest_path: Path,
    split_name: str,
) -> tuple[BankingRecord, ...]:
    """Load one verified split without accessing any unrequested source split."""

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name must be train, validation or test")

    config = DatasetConfig.from_yaml(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest.get("dataset") != config.name:
        raise ValueError("manifest dataset does not match dataset configuration")
    if manifest.get("source_commit") != config.commit:
        raise ValueError("manifest source commit does not match dataset configuration")

    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, dict):
        raise ValueError("manifest sources must be a mapping")
    observed_hashes = {
        name: details.get("sha256")
        for name, details in manifest_sources.items()
        if isinstance(details, dict)
    }
    if observed_hashes != dict(config.expected_sha256):
        raise ValueError("manifest source hashes do not match dataset configuration")

    split_details = manifest.get("splits", {}).get(split_name)
    if not isinstance(split_details, dict):
        raise ValueError(f"manifest is missing split: {split_name}")
    source_name = split_details.get("source")
    if source_name not in {"official_train", "official_test"}:
        raise ValueError(f"unsupported manifest source for {split_name}")

    source_path = raw_directory / config.files[source_name]
    if not source_path.exists():
        raise FileNotFoundError(f"verified source is missing: {source_path}")
    _verify_source_hash(source_path, config.expected_sha256[source_name])
    expected_rows = (
        config.expected_train_rows
        if source_name == "official_train"
        else config.expected_test_rows
    )
    source_records = read_banking_csv(
        source_path,
        source_split=source_name,
        expected_rows=expected_rows,
    )

    indices = split_details.get("source_indices")
    if (
        not isinstance(indices, list)
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise ValueError(f"{split_name} source indices must be unique integers")
    if split_details.get("source_indices_sha256") != stable_json_sha256(indices):
        raise ValueError(f"{split_name} source-index hash check failed")
    if split_details.get("count") != len(indices):
        raise ValueError(f"{split_name} count does not match its source indices")
    if any(index < 0 or index >= len(source_records) for index in indices):
        raise ValueError(f"{split_name} contains an out-of-range source index")

    selected = tuple(source_records[index] for index in indices)
    labels = manifest.get("label_names")
    if not isinstance(labels, list) or len(labels) != config.expected_label_count:
        raise ValueError("manifest label names do not match dataset configuration")
    if _label_distribution(selected, labels) != split_details.get("label_distribution"):
        raise ValueError(f"{split_name} label distribution check failed")
    return selected


def finite_fraction(value: float) -> float:
    """Small validation helper kept public for CLI configuration checks."""

    if not math.isfinite(value) or not 0 < value < 0.5:
        raise ValueError("fraction must be finite and between 0 and 0.5")
    return value
