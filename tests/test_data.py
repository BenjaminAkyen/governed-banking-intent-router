from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from governed_banking.data import (
    NORMALIZATION_VERSION,
    BankingRecord,
    DatasetConfig,
    acquire_and_prepare,
    build_manifest,
    load_manifest_split,
    normalize_text,
    prepare_splits,
    read_banking_csv,
    validate_manifest,
)


def _records(split: str, rows: list[tuple[str, str]]) -> tuple[BankingRecord, ...]:
    return tuple(
        BankingRecord(split, index, text, label)
        for index, (text, label) in enumerate(rows)
    )


def _synthetic_sources() -> tuple[tuple[BankingRecord, ...], tuple[BankingRecord, ...]]:
    train_rows = [
        (f"alpha request {index}", "alpha") for index in range(12)
    ] + [(f"beta request {index}", "beta") for index in range(12)]
    train_rows.extend(
        [
            ("Repeated alpha request", "alpha"),
            (" repeated   ALPHA request ", "alpha"),
            ("Conflicting request", "alpha"),
            ("conflicting request", "beta"),
        ]
    )
    test_rows = [
        ("alpha request 0", "alpha"),
        ("new alpha test", "alpha"),
        ("new beta test", "beta"),
    ]
    return _records("official_train", train_rows), _records("official_test", test_rows)


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "category"])
        writer.writerows(rows)


def test_normalize_text_is_conservative_and_repeatable() -> None:
    assert normalize_text("  CARD\u00a0Arrival  ") == "card arrival"
    assert normalize_text("Payment, reversed?") == "payment, reversed?"


def test_config_pins_a_full_commit_and_expected_dataset_shape() -> None:
    config = DatasetConfig.from_yaml(Path("configs/dataset.yaml"))

    assert config.commit == "57ec275d8078af65b7731c2a98be812d844a6d6b"
    assert config.expected_train_rows == 10_003
    assert config.expected_test_rows == 3_080
    assert config.expected_label_count == 77
    assert config.normalization == NORMALIZATION_VERSION
    assert config.expected_sha256["official_train"] == (
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
    )
    assert config.source_url("official_train").endswith(f"/{config.commit}/banking_data/train.csv")


def test_committed_manifest_matches_the_registered_data_contract() -> None:
    config = DatasetConfig.from_yaml(Path("configs/dataset.yaml"))
    manifest = json.loads(
        Path("data/manifests/banking77-seed-42.json").read_text(encoding="utf-8")
    )

    validate_manifest(manifest)
    assert manifest["source_commit"] == config.commit
    assert {
        name: details["sha256"] for name, details in manifest["sources"].items()
    } == config.expected_sha256
    assert manifest["splits"]["train"]["count"] == 8_495
    assert manifest["splits"]["validation"]["count"] == 1_501
    assert manifest["splits"]["test"]["source_indices"] == list(range(3_080))
    assert len(manifest["quarantined_train"]) == 7
    assert manifest["integrity"]["final_train_test_overlap_groups"] == 0


def test_group_split_quarantines_leakage_and_label_conflicts() -> None:
    official_train, official_test = _synthetic_sources()

    splits = prepare_splits(
        official_train,
        official_test,
        ("alpha", "beta"),
        seed=42,
        validation_fraction=0.25,
    )

    train_keys = {row.normalized_text for row in splits.train}
    validation_keys = {row.normalized_text for row in splits.validation}
    test_keys = {row.normalized_text for row in splits.test}
    assert not train_keys & validation_keys
    assert not train_keys & test_keys
    assert not validation_keys & test_keys
    assert splits.test == official_test
    assert splits.original_cross_split_overlap_groups == 1
    assert splits.original_conflicting_train_groups == 1
    assert {row.reason for row in splits.quarantined_train} == {
        "conflicting_training_labels",
        "duplicates_official_test",
    }

    repeated_locations = {
        "train": sum(row.normalized_text == "repeated alpha request" for row in splits.train),
        "validation": sum(
            row.normalized_text == "repeated alpha request" for row in splits.validation
        ),
    }
    assert sorted(repeated_locations.values()) == [0, 2]


def test_group_split_is_deterministic() -> None:
    official_train, official_test = _synthetic_sources()

    first = prepare_splits(
        official_train,
        official_test,
        ("alpha", "beta"),
        seed=17,
        validation_fraction=0.25,
    )
    second = prepare_splits(
        official_train,
        official_test,
        ("alpha", "beta"),
        seed=17,
        validation_fraction=0.25,
    )

    assert [row.source_index for row in first.train] == [row.source_index for row in second.train]
    assert [row.source_index for row in first.validation] == [
        row.source_index for row in second.validation
    ]


def test_csv_reader_preserves_embedded_newlines(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    _write_csv(path, [("first line\nsecond line", "alpha"), ("plain", "beta")])

    records = read_banking_csv(path, source_split="official_train", expected_rows=2)

    assert records[0].text == "first line\nsecond line"
    assert records[1].source_index == 1


def test_manifest_contains_indices_and_hashes_but_no_message_text(tmp_path: Path) -> None:
    official_train, official_test = _synthetic_sources()
    splits = prepare_splits(
        official_train,
        official_test,
        ("alpha", "beta"),
        seed=42,
        validation_fraction=0.25,
    )
    config = DatasetConfig(
        name="banking77-test",
        repository_slug="example/dataset",
        commit="a" * 40,
        subdirectory="data",
        files={
            "categories": "categories.json",
            "official_train": "train.csv",
            "official_test": "test.csv",
        },
        expected_sha256={
            "categories": "a" * 64,
            "official_train": "b" * 64,
            "official_test": "c" * 64,
        },
        expected_train_rows=len(official_train),
        expected_test_rows=len(official_test),
        expected_label_count=2,
        validation_fraction=0.25,
        seed=42,
        normalization=NORMALIZATION_VERSION,
    )
    source_paths = {}
    for logical_name, filename in config.files.items():
        path = tmp_path / filename
        path.write_text(f"source for {logical_name}", encoding="utf-8")
        source_paths[logical_name] = path

    manifest = build_manifest(config, source_paths, splits)
    encoded = json.dumps(manifest)

    validate_manifest(manifest)
    assert "alpha request" not in encoded
    assert manifest["contains_message_text"] is False
    assert manifest["policy"]["official_test_untouched"] is True
    assert len(manifest["sources"]["official_train"]["sha256"]) == 64

    tampered = dict(manifest)
    tampered["dataset"] = "changed"
    with pytest.raises(ValueError, match="content hash"):
        validate_manifest(tampered)


def test_offline_end_to_end_preparation(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    categories = ["alpha", "beta"]
    (raw_directory / "categories.json").write_text(json.dumps(categories), encoding="utf-8")
    train_rows = [
        (f"{label} training {index}", label)
        for label in categories
        for index in range(10)
    ]
    test_rows = [(f"{label} testing", label) for label in categories]
    _write_csv(raw_directory / "train.csv", train_rows)
    _write_csv(raw_directory / "test.csv", test_rows)
    source_hashes = {
        name: hashlib.sha256((raw_directory / filename).read_bytes()).hexdigest()
        for name, filename in {
            "categories": "categories.json",
            "official_train": "train.csv",
            "official_test": "test.csv",
        }.items()
    }

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "name: banking77-test",
                "source:",
                "  repository_slug: example/dataset",
                f"  commit: {'b' * 40}",
                "  subdirectory: data",
                "  files:",
                "    categories: categories.json",
                "    official_train: train.csv",
                "    official_test: test.csv",
                "  sha256:",
                f"    categories: {source_hashes['categories']}",
                f"    official_train: {source_hashes['official_train']}",
                f"    official_test: {source_hashes['official_test']}",
                "expected:",
                "  official_train_rows: 20",
                "  official_test_rows: 2",
                "  label_count: 2",
                "split:",
                "  validation_fraction: 0.2",
                "  seed: 42",
                f"  normalization: {NORMALIZATION_VERSION}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = acquire_and_prepare(
        config_path,
        raw_directory,
        manifest_path,
        offline=True,
    )

    assert manifest_path.exists()
    assert manifest["splits"]["train"]["count"] == 16
    assert manifest["splits"]["validation"]["count"] == 4
    assert manifest["splits"]["test"]["count"] == 2
    validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))

    loaded_train = load_manifest_split(
        config_path, raw_directory, manifest_path, "train"
    )
    assert len(loaded_train) == 16
    (raw_directory / "test.csv").unlink()
    loaded_validation = load_manifest_split(
        config_path, raw_directory, manifest_path, "validation"
    )
    assert len(loaded_validation) == 4

    with (raw_directory / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered,alpha\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        acquire_and_prepare(config_path, raw_directory, manifest_path, offline=True)
