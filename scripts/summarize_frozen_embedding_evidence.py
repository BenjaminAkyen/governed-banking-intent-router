#!/usr/bin/env python3
"""Publish text-free extraction evidence from local frozen-embedding cache metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from governed_banking.baseline import assert_text_free_artifact, write_json_artifact
from governed_banking.data import sha256_file, stable_json_sha256


def main() -> None:
    cache_directory = Path("artifacts/embeddings/frozen-roberta")
    summaries: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        metadata = json.loads(
            (cache_directory / f"{split_name}-metadata.json").read_text(encoding="utf-8")
        )
        metadata_hash = metadata["cache_metadata_sha256"]
        body = dict(metadata)
        body.pop("cache_metadata_sha256")
        if stable_json_sha256(body) != metadata_hash:
            raise ValueError(f"{split_name} embedding metadata hash check failed")
        assert_text_free_artifact(metadata)
        summaries[split_name] = {
            "cache_metadata_sha256": metadata_hash,
            "rows": metadata["row_count"],
            "device": metadata["device"],
            "extraction_seconds": metadata["extraction_seconds"],
            "rows_per_second": metadata["rows_per_second"],
            "token_length": metadata["token_length"],
            "total_encoder_parameters": metadata["total_encoder_parameters"],
            "trainable_encoder_parameters": metadata["trainable_encoder_parameters"],
            "arrays": metadata["arrays"],
        }

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "frozen_embedding_extraction_evidence",
        "contains_message_text": False,
        "encoder_repository": "FacebookAI/roberta-base",
        "encoder_revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "implementation_sha256": {
            "summarize_frozen_embedding_evidence.py": sha256_file(Path(__file__))
        },
        "pooling_outputs": ["cls", "mean"],
        "splits": summaries,
    }
    artifact["evidence_sha256"] = stable_json_sha256(artifact)
    write_json_artifact(
        artifact,
        Path("reports/frozen-roberta/embedding-extraction.json"),
    )
    print(json.dumps({name: value["rows_per_second"] for name, value in summaries.items()}))


if __name__ == "__main__":
    main()
