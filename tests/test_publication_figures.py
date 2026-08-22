import json
from pathlib import Path

import pytest

from scripts.generate_publication_figures import EvidenceError, generate_publication_figures


def test_publication_figures_are_generated_from_registered_evidence(tmp_path: Path) -> None:
    manifest = generate_publication_figures(tmp_path, output_formats=("svg",))

    assert manifest["artifact_type"] == "publication_figure_manifest"
    assert manifest["contains_message_text"] is False
    assert len(manifest["figures"]) == 5
    assert manifest["derived_values"]["historical_test_macro_f1"] == {
        "tfidf_word_char_logreg": pytest.approx(0.9053010357),
        "frozen_roberta_logreg": pytest.approx(0.8964173425),
        "original_lora_roberta": pytest.approx(0.8202064851),
    }
    assert manifest["derived_values"]["calibration_assessment_means"][
        "expected_calibration_error"
    ] == {
        "raw": pytest.approx(0.0469595169),
        "temperature_scaled": pytest.approx(0.0280082064),
    }

    manifest_on_disk = json.loads(
        (tmp_path / "publication-figures-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_on_disk["source_reports"] == manifest["source_reports"]
    for figure in manifest["figures"]:
        assert len(figure["outputs"]) == 1
        output = tmp_path / Path(figure["outputs"][0]["path"]).name
        assert output.is_file()
        assert output.stat().st_size > 1_000


def test_publication_generation_rejects_unsupported_formats(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="unsupported output formats"):
        generate_publication_figures(tmp_path, output_formats=("pdf",))
