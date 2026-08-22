#!/usr/bin/env python3
"""Generate publication-ready figures from committed, metadata-only evidence reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs/images"
REPOSITORY_URL = "github.com/BenjaminAkyen/governed-banking-intent-router"

INK = "#17242B"
MUTED = "#64748B"
GRID = "#DCE5E8"
PAPER = "#FFFFFF"
SOFT = "#F4F7F6"
GREEN = "#2F855A"
BLUE = "#3E6B89"
ORANGE = "#D97745"
RED = "#C2413A"
PURPLE = "#6D5BD0"

SOURCE_REPORTS = {
    "tfidf": PROJECT_ROOT / "reports/baseline/tfidf-logreg-test.json",
    "frozen_roberta": PROJECT_ROOT / "reports/frozen-roberta/test.json",
    "lora_roberta": PROJECT_ROOT / "reports/lora-roberta/test.json",
    "calibration": PROJECT_ROOT / "reports/calibration/temperature-scaling-aggregate.json",
    "uncertainty": PROJECT_ROOT / "reports/uncertainty/selective-ood-aggregate.json",
    "robustness": PROJECT_ROOT / "reports/robustness/module13-lora-mps-assessment.json",
}


class EvidenceError(ValueError):
    """Raised when a publication figure cannot be supported by registered evidence."""


@dataclass(frozen=True)
class FigureAsset:
    """A generated figure and the report keys that support it."""

    stem: str
    title: str
    evidence_keys: tuple[str, ...]
    draw: Callable[[Mapping[str, dict[str, Any]]], Figure]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reports() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for key, path in SOURCE_REPORTS.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise EvidenceError(f"required evidence report is missing: {path}") from error
        except json.JSONDecodeError as error:
            raise EvidenceError(f"evidence report is not valid JSON: {path}: {error}") from error
        if not isinstance(payload, dict):
            raise EvidenceError(f"evidence report must contain a JSON object: {path}")
        reports[key] = payload
    return reports


def _nested_number(payload: Mapping[str, Any], path: Sequence[str]) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise EvidenceError(f"missing required evidence field: {'.'.join(path)}")
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise EvidenceError(f"evidence field must be numeric: {'.'.join(path)}")
    return float(current)


def _nested_bool(payload: Mapping[str, Any], path: Sequence[str]) -> bool:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise EvidenceError(f"missing required evidence field: {'.'.join(path)}")
        current = current[key]
    if not isinstance(current, bool):
        raise EvidenceError(f"evidence field must be boolean: {'.'.join(path)}")
    return current


def _assert_text_free(reports: Mapping[str, dict[str, Any]]) -> None:
    prohibited_truthy_fields = (
        "contains_message_text",
        "contains_input_text",
        "contains_redacted_text",
        "contains_message_hash",
    )
    for key, report in reports.items():
        for field in prohibited_truthy_fields:
            if report.get(field) is True:
                raise EvidenceError(f"refusing to publish text-bearing report: {key}.{field}")


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.facecolor": PAPER,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "figure.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "savefig.facecolor": PAPER,
            "svg.hashsalt": "governed-banking-intent-router-publication-v1",
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
        }
    )


def _figure_header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.93, title, fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.07, 0.875, subtitle, fontsize=12.5, color=MUTED, va="top")


def _figure_footer(fig: Figure, text: str) -> None:
    fig.text(0.07, 0.035, text, fontsize=9.5, color=MUTED, va="bottom")
    fig.text(0.93, 0.035, REPOSITORY_URL, fontsize=9.5, color=MUTED, ha="right", va="bottom")


def _draw_model_comparison(reports: Mapping[str, dict[str, Any]]) -> Figure:
    models = (
        ("TF-IDF word + character\nlogistic regression", "tfidf", GREEN),
        ("Frozen RoBERTa embeddings\n+ logistic regression", "frozen_roberta", BLUE),
        ("Original rank-8\nLoRA-RoBERTa", "lora_roberta", ORANGE),
    )
    scores = [
        _nested_number(reports[key], ("test_result", "metrics", "macro_f1")) for _, key, _ in models
    ]
    seeds = {reports[key].get("random_seed", 42) for _, key, _ in models}
    if seeds != {42}:
        raise EvidenceError(f"historical comparison requires seed 42 reports; observed {seeds}")

    fig, axis = plt.subplots(figsize=(13.4, 7.3))
    fig.subplots_adjust(left=0.31, right=0.93, top=0.78, bottom=0.16)
    _figure_header(
        fig,
        "The simplest model won the historical comparison",
        "Macro-F1 on the previously observed official BANKING77 test split · "
        "seed 42 · higher is better",
    )
    labels = [label for label, _, _ in models]
    colors = [color for _, _, color in models]
    y_positions = list(range(len(models)))
    bars = axis.barh(y_positions, scores, color=colors, height=0.52, zorder=3)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    axis.tick_params(axis="y", length=0, pad=14)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, score in zip(bars, scores, strict=True):
        axis.text(
            score + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=INK,
        )
    axis.text(
        scores[0] - 0.015,
        bars[0].get_y() + bars[0].get_height() / 2,
        "CHAMPION RETAINED",
        va="center",
        ha="right",
        fontsize=9.5,
        fontweight="bold",
        color=PAPER,
    )
    _figure_footer(
        fig,
        "Research preview · single-seed historical test evidence · "
        "not a fresh promotion evaluation",
    )
    return fig


def _draw_calibration(reports: Mapping[str, dict[str, Any]]) -> Figure:
    report = reports["calibration"]
    if not _nested_bool(report, ("data_boundary", "fit_and_assessment_rows_disjoint")):
        raise EvidenceError("calibration figure requires disjoint fit and assessment roles")
    metrics = (
        ("Calibration error (ECE)", "expected_calibration_error", "ECE"),
        ("Log loss (NLL)", "negative_log_likelihood", "NLL"),
        ("Brier score", "multiclass_brier_score", "Brier"),
    )
    values: list[tuple[float, float]] = []
    for _, key, _ in metrics:
        raw = _nested_number(report, ("assessment_metrics", key, "raw", "mean"))
        scaled = _nested_number(report, ("assessment_metrics", key, "calibrated", "mean"))
        if scaled >= raw:
            raise EvidenceError(f"registered calibrated {key} is not lower than raw")
        values.append((raw, scaled))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 7.3))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.75, bottom=0.18, wspace=0.38)
    _figure_header(
        fig,
        "Temperature scaling improved calibration",
        "Means across seeds 17, 42 and 73 · calibration-fit and assessment rows are "
        "disjoint · lower is better",
    )
    for axis, (label, _, short_label), (raw, scaled) in zip(axes, metrics, values, strict=True):
        axis.barh([1, 0], [raw, scaled], color=["#BAC6CB", GREEN], height=0.48)
        limit = max(raw, scaled) * 1.32
        axis.set_xlim(0, limit)
        axis.set_ylim(-0.7, 1.75)
        axis.set_yticks([1, 0], ["Raw", "Scaled"])
        axis.set_title(label, loc="left", fontsize=12.5, fontweight="bold", pad=15)
        axis.grid(axis="x", color=GRID, linewidth=1, zorder=0)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelbottom=False, bottom=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(raw + limit * 0.025, 1, f"{raw:.4f}", va="center", fontweight="bold")
        axis.text(scaled + limit * 0.025, 0, f"{scaled:.4f}", va="center", fontweight="bold")
        reduction = (raw - scaled) / raw
        axis.text(
            0,
            -0.53,
            f"{short_label}: {reduction:.1%} lower",
            fontsize=11,
            fontweight="bold",
            color=GREEN,
        )
    _figure_footer(
        fig,
        "Post-selection, post-test exploratory evidence · supports calibration claims only",
    )
    return fig


def _common_uncertainty_thresholds(report: Mapping[str, Any]) -> tuple[float, float, float]:
    by_seed = report.get("acceptance_gate", {}).get("by_seed")
    if not isinstance(by_seed, Mapping) or not by_seed:
        raise EvidenceError("uncertainty report has no registered seed gates")
    keys = (
        "registered_minimum_known_coverage",
        "registered_maximum_selective_risk",
        "registered_minimum_possible_ood_recall",
    )
    values: list[float] = []
    for key in keys:
        observed = {float(seed_gate[key]) for seed_gate in by_seed.values() if key in seed_gate}
        if len(observed) != 1:
            raise EvidenceError(f"uncertainty threshold is missing or inconsistent: {key}")
        values.append(observed.pop())
    return values[0], values[1], values[2]


def _draw_uncertainty_gates(reports: Mapping[str, dict[str, Any]]) -> Figure:
    report = reports["uncertainty"]
    coverage_minimum, risk_maximum, ood_minimum = _common_uncertainty_thresholds(report)
    rows = (
        (
            "Known-request coverage",
            _nested_number(report, ("assessment_metrics", "known_coverage", "mean")),
            coverage_minimum,
            "minimum",
        ),
        (
            "Selective risk",
            _nested_number(report, ("assessment_metrics", "selective_risk", "mean")),
            risk_maximum,
            "maximum",
        ),
        (
            "Synthetic possible-OOD recall",
            _nested_number(report, ("assessment_metrics", "possible_ood_recall", "mean")),
            ood_minimum,
            "minimum",
        ),
    )
    statuses = [
        value >= threshold if direction == "minimum" else value <= threshold
        for _, value, threshold, direction in rows
    ]
    if _nested_bool(report, ("acceptance_gate", "all_seeds_passed")):
        raise EvidenceError("committed uncertainty report unexpectedly records all seeds passing")

    fig, axis = plt.subplots(figsize=(13.4, 7.3))
    fig.subplots_adjust(left=0.31, right=0.93, top=0.76, bottom=0.17)
    _figure_header(
        fig,
        "Uncertainty gates failed despite high AUROC",
        "Mean locked-assessment results across three seeds · registered gates were not relaxed "
        "after evaluation",
    )
    y_positions = list(range(len(rows)))
    colors = [GREEN if passed else RED for passed in statuses]
    axis.barh(y_positions, [value for _, value, _, _ in rows], height=0.38, color=colors, zorder=3)
    for y, (_, value, threshold, direction), passed in zip(
        y_positions, rows, statuses, strict=True
    ):
        axis.vlines(threshold, y - 0.31, y + 0.31, color=INK, linewidth=2, zorder=4)
        axis.text(
            min(value + 0.018, 0.94),
            y,
            f"{value:.2%}  {'PASS' if passed else 'FAIL'}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        comparator = "≥" if direction == "minimum" else "≤"
        axis.text(
            threshold,
            y + 0.43,
            f"gate {comparator} {threshold:.0%}",
            ha="center",
            fontsize=9.5,
            color=MUTED,
        )
    axis.set_yticks(y_positions, [label for label, _, _, _ in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axis.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    axis.tick_params(axis="y", length=0, pad=14)
    for spine in axis.spines.values():
        spine.set_visible(False)
    auroc = _nested_number(report, ("assessment_metrics", "possible_ood_auroc", "mean"))
    _figure_footer(
        fig,
        f"Possible-OOD AUROC {auroc:.4f} · synthetic OOD · experimental review metadata only",
    )
    return fig


def _draw_robustness_gates(reports: Mapping[str, dict[str, Any]]) -> Figure:
    report = reports["robustness"]
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvidenceError("robustness report has no metrics object")
    rows = (
        ("Acceptable intent", "in_scope_acceptable_intent_rate", 0.80),
        ("Security routing recall", "expected_security_routing_recall", 1.00),
        ("Routing-action agreement", "routing_action_match_rate", 0.95),
        ("PII expectation agreement", "pii_expectation_match_rate", 1.00),
    )
    values = [_nested_number(report, ("metrics", key)) for _, key, _ in rows]
    statuses = [value >= threshold for value, (_, _, threshold) in zip(values, rows, strict=True)]

    fig, axis = plt.subplots(figsize=(13.4, 7.3))
    fig.subplots_adjust(left=0.29, right=0.93, top=0.76, bottom=0.17)
    _figure_header(
        fig,
        "Robustness testing exposed deployment blockers",
        "60 project-authored cases · LoRA research service on Apple MPS · "
        "failure-discovery evidence",
    )
    y_positions = list(range(len(rows)))
    colors = [GREEN if passed else RED for passed in statuses]
    axis.barh(y_positions, values, color=colors, height=0.42, zorder=3)
    for y, value, (_, _, threshold), passed in zip(
        y_positions, values, rows, statuses, strict=True
    ):
        axis.vlines(threshold, y - 0.3, y + 0.3, color=INK, linewidth=2, zorder=4)
        label_x = value - 0.015 if value > 0.18 else value + 0.015
        alignment = "right" if value > 0.18 else "left"
        label_color = PAPER if value > 0.18 else INK
        axis.text(
            label_x,
            y,
            f"{value:.2%}  {'PASS' if passed else 'FAIL'}",
            va="center",
            ha=alignment,
            fontsize=11,
            fontweight="bold",
            color=label_color,
        )
        axis.text(
            threshold,
            y + 0.38,
            f"required ≥ {threshold:.0%}",
            ha="center",
            fontsize=9.2,
            color=MUTED,
        )
    axis.set_yticks(y_positions, [label for label, _, _ in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axis.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    axis.tick_params(axis="y", length=0, pad=14)
    for spine in axis.spines.values():
        spine.set_visible(False)
    suggestion_count = int(_nested_number(report, ("metrics", "suggest_queue_count")))
    _figure_footer(
        fig,
        f"{suggestion_count} autonomous suggestion actions · no customer data · "
        "not production validation",
    )
    return fig


def _box(
    axis: Axes, x: float, y: float, width: float, height: float, label: str, color: str
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=color,
        facecolor=PAPER,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )


def _arrow(
    axis: Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED
) -> None:
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.8, color=color)
    )


def _draw_architecture(_: Mapping[str, dict[str, Any]]) -> Figure:
    fig, axis = plt.subplots(figsize=(13.4, 7.3))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.15)
    _figure_header(
        fig,
        "Governance stays in the inference path",
        "Current research-service boundary · deterministic controls retain routing authority",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    _box(axis, 0.02, 0.57, 0.12, 0.17, "Support\nrequest", BLUE)
    _box(axis, 0.19, 0.57, 0.13, 0.17, "FastAPI\n/v1", BLUE)
    _box(axis, 0.37, 0.57, 0.13, 0.17, "PII\nredaction", PURPLE)
    _box(axis, 0.55, 0.57, 0.15, 0.17, "LoRA research\ninference", ORANGE)
    _box(axis, 0.75, 0.57, 0.20, 0.17, "Deterministic\nrisk-routing policy", GREEN)
    _arrow(axis, (0.14, 0.655), (0.19, 0.655))
    _arrow(axis, (0.32, 0.655), (0.37, 0.655))
    _arrow(axis, (0.50, 0.655), (0.55, 0.655))
    _arrow(axis, (0.70, 0.655), (0.75, 0.655))

    _box(axis, 0.55, 0.25, 0.15, 0.14, "Experimental\nuncertainty", RED)
    _box(axis, 0.76, 0.25, 0.09, 0.14, "Human\nreview", GREEN)
    _box(axis, 0.88, 0.25, 0.10, 0.14, "Security\nqueue", RED)
    _arrow(axis, (0.625, 0.57), (0.625, 0.39))
    _arrow(axis, (0.84, 0.57), (0.805, 0.39), GREEN)
    _arrow(axis, (0.88, 0.57), (0.93, 0.39), RED)

    _box(axis, 0.19, 0.25, 0.13, 0.14, "Metadata-only\naudit events", PURPLE)
    _box(axis, 0.37, 0.25, 0.13, 0.14, "Privacy-safe\ntelemetry", BLUE)
    _arrow(axis, (0.78, 0.57), (0.48, 0.39), MUTED)
    _arrow(axis, (0.75, 0.60), (0.29, 0.39), MUTED)

    axis.text(
        0.02,
        0.09,
        "Champion registry: TF-IDF retained  |  Shadow service: LoRA challenger  |  "
        "Allowed actions: human_review or security_queue",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
    )
    _figure_footer(fig, "Research preview · no autonomous customer-service action")
    return fig


FIGURES = (
    FigureAsset(
        "model-macro-f1-comparison",
        "Historical model macro-F1 comparison",
        ("tfidf", "frozen_roberta", "lora_roberta"),
        _draw_model_comparison,
    ),
    FigureAsset(
        "calibration-before-after",
        "Calibration metrics before and after temperature scaling",
        ("calibration",),
        _draw_calibration,
    ),
    FigureAsset(
        "uncertainty-gate-results",
        "Selective-prediction and possible-OOD gate results",
        ("uncertainty",),
        _draw_uncertainty_gates,
    ),
    FigureAsset(
        "robustness-gate-results",
        "Synthetic robustness gate results",
        ("robustness",),
        _draw_robustness_gates,
    ),
    FigureAsset(
        "governed-routing-architecture",
        "Governed research-service architecture",
        (),
        _draw_architecture,
    ),
)


def _save_figure(fig: Figure, output_path: Path, output_format: str) -> None:
    metadata: dict[str, Any]
    if output_format == "svg":
        metadata = {"Creator": "governed-banking-intent-router", "Date": None}
    else:
        metadata = {"Software": "governed-banking-intent-router"}
    fig.savefig(output_path, format=output_format, dpi=180, metadata=metadata)
    plt.close(fig)


def _derived_values(reports: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    calibration = reports["calibration"]
    uncertainty = reports["uncertainty"]
    robustness = reports["robustness"]
    return {
        "historical_test_macro_f1": {
            "tfidf_word_char_logreg": _nested_number(
                reports["tfidf"], ("test_result", "metrics", "macro_f1")
            ),
            "frozen_roberta_logreg": _nested_number(
                reports["frozen_roberta"], ("test_result", "metrics", "macro_f1")
            ),
            "original_lora_roberta": _nested_number(
                reports["lora_roberta"], ("test_result", "metrics", "macro_f1")
            ),
        },
        "calibration_assessment_means": {
            key: {
                "raw": _nested_number(calibration, ("assessment_metrics", key, "raw", "mean")),
                "temperature_scaled": _nested_number(
                    calibration, ("assessment_metrics", key, "calibrated", "mean")
                ),
            }
            for key in (
                "expected_calibration_error",
                "negative_log_likelihood",
                "multiclass_brier_score",
            )
        },
        "uncertainty_assessment_means": {
            key: _nested_number(uncertainty, ("assessment_metrics", key, "mean"))
            for key in (
                "known_coverage",
                "selective_risk",
                "possible_ood_recall",
                "possible_ood_auroc",
            )
        },
        "robustness_metrics": {
            key: _nested_number(robustness, ("metrics", key))
            for key in (
                "in_scope_acceptable_intent_rate",
                "expected_security_routing_recall",
                "routing_action_match_rate",
                "pii_expectation_match_rate",
                "suggest_queue_count",
            )
        },
    }


def generate_publication_figures(
    output_directory: Path,
    *,
    output_formats: Sequence[str] = ("png", "svg"),
) -> dict[str, Any]:
    """Generate every registered publication figure and return its evidence manifest."""

    unsupported = set(output_formats) - {"png", "svg"}
    if unsupported:
        raise EvidenceError(f"unsupported output formats: {sorted(unsupported)}")
    if not output_formats:
        raise EvidenceError("at least one output format is required")

    reports = _load_reports()
    _assert_text_free(reports)
    _configure_style()
    output_directory.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for asset in FIGURES:
        output_records: list[dict[str, str]] = []
        for output_format in output_formats:
            figure = asset.draw(reports)
            output_path = output_directory / f"{asset.stem}.{output_format}"
            _save_figure(figure, output_path, output_format)
            output_records.append(
                {
                    "path": output_path.relative_to(PROJECT_ROOT).as_posix()
                    if output_path.is_relative_to(PROJECT_ROOT)
                    else output_path.name,
                    "sha256": _sha256_file(output_path),
                }
            )
        generated.append(
            {
                "id": asset.stem,
                "title": asset.title,
                "evidence": [
                    SOURCE_REPORTS[key].relative_to(PROJECT_ROOT).as_posix()
                    for key in asset.evidence_keys
                ],
                "outputs": output_records,
            }
        )

    manifest = {
        "schema_version": 1,
        "artifact_type": "publication_figure_manifest",
        "claim_scope": "research_preview",
        "contains_message_text": False,
        "generator": {
            "path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256_file(Path(__file__).resolve()),
            "matplotlib_version": matplotlib.__version__,
        },
        "source_reports": {
            SOURCE_REPORTS[key].relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
            for key, path in SOURCE_REPORTS.items()
        },
        "derived_values": _derived_values(reports),
        "figures": generated,
        "limitations": [
            "Historical BANKING77 test metrics are single-seed and previously observed.",
            "Calibration and uncertainty evidence is post-selection and post-test exploratory.",
            "Robustness evidence uses a small synthetic pack and is not production validation.",
            "Rendering hashes can differ across Matplotlib or font-library versions; source "
            "hashes and derived values preserve the evidence lineage.",
        ],
    }
    manifest_path = output_directory / "publication-figures-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="destination directory (default: docs/images)",
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("png", "svg"),
        dest="formats",
        help="output format; repeat to select multiple formats (default: PNG and SVG)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formats = tuple(args.formats) if args.formats else ("png", "svg")
    try:
        manifest = generate_publication_figures(args.output_dir.resolve(), output_formats=formats)
    except (EvidenceError, OSError) as error:
        print(f"Publication figure generation failed: {error}", file=sys.stderr)
        return 2
    print(f"Generated {len(manifest['figures'])} figures in {args.output_dir.resolve()}")
    print(f"Manifest: {(args.output_dir / 'publication-figures-manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
