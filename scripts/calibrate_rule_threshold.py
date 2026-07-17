"""Select the rule weak/detected threshold from the fixed calibration split."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import polars as pl


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config.py").exists() and (candidate / "services").is_dir():
            return candidate
    raise FileNotFoundError("Root proyek tidak ditemukan")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from services.artifact_service import ArtifactService
from services.evaluation_service import EvaluationService
from services.lexicon_sentiment_service import LexiconSentimentService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze RULE_WEAK_THRESHOLD using only the calibration split."
    )
    parser.add_argument("--dataset", type=Path, default=config.TRAINING_DATASET_WITH_SPLIT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RULE_THRESHOLD_CALIBRATION_ARTIFACT_PATH,
    )
    parser.add_argument("--text-column", default=config.COL_PROCESSED)
    parser.add_argument("--actual-column", default="sentiment_label")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--min-precision", type=float, default=config.RULE_MIN_PRECISION)
    parser.add_argument("--min-coverage", type=float, default=config.RULE_MIN_COVERAGE)
    parser.add_argument(
        "--preferred-threshold",
        type=float,
        default=config.RULE_PREFERRED_WEAK_THRESHOLD,
    )
    parser.add_argument(
        "--selection-mode",
        choices=("preferred-if-valid", "max-coverage"),
        default="preferred-if-valid",
    )
    parser.add_argument(
        "--candidates",
        type=float,
        nargs="*",
        default=list(config.RULE_WEAK_THRESHOLD_CANDIDATES),
    )
    return parser.parse_args()


def evaluate_threshold(
    df: pl.DataFrame,
    *,
    threshold: float,
    text_column: str,
    actual_column: str,
) -> dict[str, Any]:
    predicted = LexiconSentimentService(weak_threshold=threshold).analyze_dataframe(
        df,
        text_column=text_column,
    )
    detected = predicted.filter(
        pl.col(config.COL_RULE_STATUS) == config.RULE_STATUS_DETECTED
    )
    correct_detected = detected.filter(
        pl.col(config.COL_RULE_LABEL) == pl.col(actual_column)
    ).height
    detected_count = detected.height
    total = df.height
    status_distribution = dict(Counter(predicted[config.COL_RULE_STATUS].to_list()))
    metrics = EvaluationService().evaluate_predictions(
        predicted[actual_column].to_list(),
        predicted[config.COL_RULE_LABEL].to_list(),
    )
    return {
        "threshold": round(float(threshold), 4),
        "detected_count": detected_count,
        "correct_detected_count": correct_detected,
        "wrong_detected_count": detected_count - correct_detected,
        "weak_count": status_distribution.get(config.RULE_STATUS_WEAK, 0),
        "unknown_count": status_distribution.get(config.RULE_STATUS_UNKNOWN, 0),
        "precision_on_detected": round(
            correct_detected / detected_count if detected_count else 0.0,
            6,
        ),
        "coverage": round(detected_count / total if total else 0.0, 6),
        "status_distribution": status_distribution,
        "rule_label_metrics_all_rows": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
    }


def valid_candidate(
    candidate: dict[str, Any],
    *,
    min_precision: float,
    min_coverage: float,
) -> bool:
    return (
        int(candidate["detected_count"]) > 0
        and float(candidate["precision_on_detected"]) >= min_precision
        and float(candidate["coverage"]) >= min_coverage
    )


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    preferred_threshold: float,
    min_precision: float,
    min_coverage: float,
    selection_mode: str,
) -> dict[str, Any]:
    valid = [
        candidate
        for candidate in candidates
        if valid_candidate(
            candidate,
            min_precision=min_precision,
            min_coverage=min_coverage,
        )
    ]
    if not valid:
        raise RuntimeError(
            "Tidak ada RULE_WEAK_THRESHOLD yang memenuhi target calibration "
            f"precision >= {min_precision} dan coverage >= {min_coverage}"
        )

    preferred = next(
        (
            candidate
            for candidate in valid
            if abs(float(candidate["threshold"]) - preferred_threshold) < 1e-9
        ),
        None,
    )
    if selection_mode == "preferred-if-valid" and preferred is not None:
        return preferred

    return sorted(
        valid,
        key=lambda item: (
            -float(item["coverage"]),
            -float(item["precision_on_detected"]),
            float(item["threshold"]),
        ),
    )[0]


def main() -> None:
    args = parse_args()
    artifact = ArtifactService()
    df = pl.read_parquet(args.dataset)
    required = {args.text_column, args.actual_column, args.split_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Kolom dataset hilang: {missing}")

    calibration_df = df.filter(pl.col(args.split_column) == args.calibration_split)
    if calibration_df.is_empty():
        raise SystemExit("Calibration split kosong; threshold rule tidak dapat dipilih")

    candidates = sorted({round(float(value), 4) for value in args.candidates})
    evaluations = [
        evaluate_threshold(
            calibration_df,
            threshold=threshold,
            text_column=args.text_column,
            actual_column=args.actual_column,
        )
        for threshold in candidates
    ]
    selected = select_candidate(
        evaluations,
        preferred_threshold=float(args.preferred_threshold),
        min_precision=float(args.min_precision),
        min_coverage=float(args.min_coverage),
        selection_mode=args.selection_mode,
    )

    payload: dict[str, Any] = {
        "purpose": "Freeze RULE_WEAK_THRESHOLD before fixed test evaluation.",
        "test_usage_policy": (
            "Only rows with split == calibration are used for threshold selection; "
            "test rows are not used for threshold, weight, or policy decisions."
        ),
        "dataset": {
            "path": str(args.dataset),
            "sha256": artifact.file_sha256(args.dataset) if args.dataset.exists() else None,
            "split_column": args.split_column,
            "split_used": args.calibration_split,
            "total_rows_in_file": df.height,
            "calibration_rows": calibration_df.height,
            "label_distribution": dict(Counter(calibration_df[args.actual_column].to_list())),
        },
        "selection_policy": {
            "mode": args.selection_mode,
            "preferred_threshold": float(args.preferred_threshold),
            "min_precision_on_detected": float(args.min_precision),
            "min_coverage": float(args.min_coverage),
            "fallback": "max coverage, then precision, then lower threshold",
        },
        "selected": selected,
        "candidate_count": len(evaluations),
        "candidates": evaluations,
        "final_config": {
            "RULE_WEAK_THRESHOLD": selected["threshold"],
            "RULE_MIN_PRECISION": float(args.min_precision),
            "RULE_MIN_COVERAGE": float(args.min_coverage),
        },
    }
    artifact.save_json(payload, args.output)
    print(f"Selected RULE_WEAK_THRESHOLD: {selected['threshold']:.2f}")
    print(
        "Calibration precision-on-detected: "
        f"{selected['precision_on_detected']:.4f}"
    )
    print(f"Calibration coverage: {selected['coverage']:.4f}")
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()
